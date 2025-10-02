// ===================== Batched Attention Kernels =====================

// ---------- k_attn_scores_batch ----------
// qB:        [B, n_heads*head_dim]
// k_layer:   [B, seq_len, kv_dim]   (đã offset đến layer l)  kv_dim = head_dim*n_kv
// mask:      [seq_len, seq_len]     hoặc nullptr
// attB:      [B, n_heads, row_stride]  (row_stride = seq_len+1)
// positions: [B] (device)
// Tính att[b,h,t] cho t = 0..pos[b] (CHƯA có sink)
// ---------- scores ----------
__global__ void k_attn_scores_batch_bf16k(
    const float*        __restrict__ qB,        // [B, n_heads, head_dim] (packed)
    const hip_bfloat16* __restrict__ k_layer,   // base pointer đã offset tới layer hiện tại
    const float*        __restrict__ mask,      // [seq_len, seq_len] hoặc [pos, t] flatten (row-major), có thể null
    float*              __restrict__ attB,      // [B, n_heads, row_stride]
    const int*          __restrict__ positions, // [B] vị trí tuyệt đối (pos_abs)
    int head_dim, int kv_mul,
    int seq_len,         // chiều logic của attention hàng (row_stride logic)
    int kv_dim,          // = (n_kv * head_dim)
    int n_heads,
    int row_stride,      // nên = seq_len
    int B,
    int sliding_window,  // W; nếu 0 thì coi như full
    int len_this,        // chiều lưu trữ theo time cho layer hiện tại: = (layer lẻ ? seq_len : max(W,1))
    int use_ring         // 1 nếu layer hiện tại dùng ring (even & W>0), ngược lại 0
)
{
  const int h = blockIdx.x; // 0..n_heads-1
  const int b = blockIdx.y; // 0..B-1
  if (h >= n_heads || b >= B) return;

  // Cache Q vào shared memory để tránh đọc lặp lại
  extern __shared__ float sQ[]; // size = head_dim * sizeof(float)
  const float* qh = qB + (b * n_heads + h) * head_dim;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    sQ[i] = qh[i];
  }
  __syncthreads();

  const int  kv_head = h / kv_mul;         // map Q-head -> K-head (MHA / MQA)
  const int  pos_abs = positions[b];       // vị trí tuyệt đối
  const int  W       = sliding_window > 0 ? sliding_window : 0;
  const int  start   = (W > 0) ? max(0, pos_abs - W + 1) : 0;
  const bool has_mask = (mask != nullptr);
  const float inv_sqrt_d = rsqrtf((float)head_dim);

  // Row output cho (b,h)
  float* att_row = attB + (b * n_heads + h) * row_stride;

  // 1) Fill các cột bị loại (t < start) mà không cần dot (tiết kiệm compute & DRAM)
  if (W > 0) {
    for (int t = threadIdx.x; t < start; t += blockDim.x) {
      // nếu có mask cho toàn hàng pos_abs, dùng luôn để đảm bảo -inf chính xác
      att_row[t] = has_mask ? mask[pos_abs * seq_len + t]
                            : -INFINITY;  // fallback an toàn nếu không truyền mask
    }
  }

  // 2) Tính dot cho vùng hữu ích: t ∈ [start, pos_abs]
  for (int t = start + threadIdx.x; t <= pos_abs; t += blockDim.x) {
    // Slot lưu trữ trong K: nếu ring thì slot = t % len_this, else slot = t
    const int slot = use_ring ? (t % len_this) : t;

    // Base K-row: [b, slot, kv_head, :head_dim]
    const hip_bfloat16* kt = k_layer
      + ((b * len_this + slot) * kv_dim)
      +  (kv_head * head_dim);

    // dot(sQ, kt_bf16) → FP32
    float s = 0.f;

    // Unroll nhẹ để compiler vectorize (độc lập head_dim % 4)
    int i = 0;
    #pragma unroll 4
    for (; i + 4 <= head_dim; i += 4) {
      float k0 = bf16_to_f32(kt[i + 0]);
      float k1 = bf16_to_f32(kt[i + 1]);
      float k2 = bf16_to_f32(kt[i + 2]);
      float k3 = bf16_to_f32(kt[i + 3]);

      s = fmaf(sQ[i + 0], k0, s);
      s = fmaf(sQ[i + 1], k1, s);
      s = fmaf(sQ[i + 2], k2, s);
      s = fmaf(sQ[i + 3], k3, s);
    }
    for (; i < head_dim; ++i) {
      s = fmaf(sQ[i], bf16_to_f32(kt[i]), s);
    }

    s *= inv_sqrt_d;
    if (has_mask) {
      s += mask[pos_abs * seq_len + t];
    }
    att_row[t] = s;
  }
}

static inline void attn_scores_gpu_batch_bf16k(
    const float*        qB,             // [B, n_heads, head_dim]
    const hip_bfloat16* k_all,          // KV cache tổng cho tất cả layer
    const float*        mask,           // có thể null; nếu dùng SW nên truyền bảng mask[pos, t]
    float*              attB,           // [B, n_heads, seq_len]
    const int*          d_positions,    // [B]
    int head_dim, int kv_mul,
    int seq_len, int kv_dim,
    int n_heads, int B,
    int layer,              // <--- thêm: để wrapper tính offset & len_this
    int MAX_BATCH_SIZE,
    int sliding_window,     // = W (0 nếu tắt)
    hipStream_t stream = 0)
{
  // Xác định len lưu trữ cho layer này
  const int len_even = (sliding_window > 0 ? sliding_window : seq_len);
  const bool is_even_layer = ((layer % 2) == 0);
  const int  len_this = is_even_layer ? len_even : seq_len;
  const int  use_ring = (is_even_layer && sliding_window > 0) ? 1 : 0;

  // Kích thước khối cho odd/even layer (tính theo phần tử hip_bfloat16)
  const size_t odd_block = MAX_BATCH_SIZE * seq_len    * kv_dim;
  const size_t even_block= MAX_BATCH_SIZE * len_even   * kv_dim;

  // Số layer trước đó theo parity (như các hàm bạn đã dùng)
  const size_t evens_before = ((layer + 1) >> 1);
  const size_t odds_before  = (layer >> 1);

  // Offset base của layer hiện tại trong k_all
  const size_t layer_base = evens_before * even_block + odds_before * odd_block;
  const hip_bfloat16* k_layer = k_all + layer_base;

  // Cấu hình kernel
  const int BS = 256;
  dim3 block(BS);
  dim3 grid(n_heads, B, 1);
  const size_t shmem_bytes = head_dim * sizeof(float); // cache Q

  hipLaunchKernelGGL(
    k_attn_scores_batch_bf16k,
    grid, block, shmem_bytes, stream,
    qB, k_layer, mask, attB, d_positions,
    head_dim, kv_mul,
    seq_len, kv_dim,                // seq_len: chiều logic của hàng attention
    n_heads,
    /*row_stride=*/seq_len + 1,         // để đơn giản & đồng bộ với mask
    B,
    layer % 2 ? 0 : sliding_window,
    len_this, use_ring
  );
}

// Giả sử đã có:
//   - inline float bf16_to_f32(hip_bfloat16 x);
//   - inline hip_bfloat16 f32_to_bf16(float x);

__global__ void k_attn_weighted_sum_batch_bf16v(
    const float*        __restrict__ attB,      // [B, n_heads, row_stride=seq_len]
    const hip_bfloat16* __restrict__ v_layer,   // base pointer đã offset tới layer hiện tại
    hip_bfloat16*       __restrict__ tbB,       // [B, n_heads, head_dim]
    const int*          __restrict__ positions, // [B] pos tuyệt đối
    int head_dim, int kv_mul,
    int row_stride,      // = seq_len
    int seq_len,
    int kv_dim,          // = n_kv * head_dim
    int n_heads, int B,
    int sliding_window,  // = W (0 nếu tắt)
    int len_this,        // = (layer lẻ ? seq_len : max(W,1))
    int use_ring         // 1 nếu even-layer & W>0; ngược lại 0
)
{
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  if (h >= n_heads || b >= B) return;

  const int pos_abs = positions[b];
  const int W       = (sliding_window > 0 ? sliding_window : 0);
  const int start   = (W > 0 ? max(0, pos_abs - W + 1) : 0);

  const int kv_head = h / kv_mul;

  const float* att_row = attB + (b * n_heads + h) * row_stride;
  hip_bfloat16* out_h  = tbB   + (b * n_heads + h) * head_dim;

  // Sẵn sàng base offset cho batch & head trong V
  const size_t batch_base = b * len_this * kv_dim;
  const size_t head_base  = kv_head * head_dim;

  // slot tuần tự (đỡ tốn phép % mỗi vòng)
  int slot = use_ring ? (start % len_this) : start;

  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float acc = 0.f;

    // Duyệt vùng hữu ích [start..pos_abs]
    for (int t = start; t <= pos_abs; ++t) {
      const float a = att_row[t];

      const size_t row_off = batch_base + slot * kv_dim + head_base;
      const hip_bfloat16* vt = v_layer + row_off; // vt[0..head_dim-1]

      acc = fmaf(a, bf16_to_f32(vt[i]), acc);

      if (use_ring) {
        ++slot; if (slot == len_this) slot = 0;
      } else {
        ++slot; // = t+1
      }
    }

    out_h[i] = f32_to_bf16(acc);

    // Khôi phục slot cho thread này nếu còn làm i tiếp theo
    // (vì mỗi i cần lại duyệt từ start..pos_abs)
    if (use_ring) {
      // trở về slot của 'start'
      slot = start % len_this;
    } else {
      slot = start;
    }
  }
}

static inline void attn_weighted_sum_gpu_batch_bf16v(
    const float*        attB,            // [B, n_heads, seq_len]
    const hip_bfloat16* v_all,           // KV cache tổng cho tất cả layer
    hip_bfloat16*       tbB,             // [B, n_heads, head_dim]
    const int*          d_positions,     // [B]
    int head_dim, int kv_mul,
    int seq_len, int kv_dim,
    int n_heads, int B,
    int layer,               // <--- thêm: để wrapper tính offset & len_this
    int MAX_BATCH_SIZE,
    int sliding_window,      // = W (0 nếu tắt)
    hipStream_t stream = 0)
{
  const int len_even = (sliding_window > 0 ? sliding_window : seq_len);
  const bool is_even_layer = ((layer % 2) == 0);
  const int  len_this = is_even_layer ? len_even : seq_len;
  const int  use_ring = (is_even_layer && sliding_window > 0) ? 1 : 0;

  // Kích thước block theo parity (tính theo phần tử hip_bfloat16)
  const size_t odd_block  = MAX_BATCH_SIZE * seq_len  * kv_dim;
  const size_t even_block = MAX_BATCH_SIZE * len_even * kv_dim;

  // Số layer trước đó theo parity (như bạn đang dùng ở các hàm khác)
  const size_t evens_before = ((layer + 1) >> 1);
  const size_t odds_before  = (layer >> 1);

  const size_t layer_base = evens_before * even_block + odds_before * odd_block;
  const hip_bfloat16* v_layer = v_all + layer_base;

  const int BS = 256;
  dim3 block(BS);
  dim3 grid(n_heads, B, 1);

  hipLaunchKernelGGL(
    k_attn_weighted_sum_batch_bf16v,
    grid, block, /*shmem=*/0, stream,
    attB, v_layer, tbB, d_positions,
    head_dim, kv_mul,
    /*row_stride=*/seq_len + 1, /*seq_len=*/seq_len,
    kv_dim, n_heads, B,
    layer % 2 ? 0 : sliding_window, len_this, use_ring
  );
}
