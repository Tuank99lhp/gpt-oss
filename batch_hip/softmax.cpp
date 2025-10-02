// ===================== Batched Attention Kernels =====================

// ---------- k_softmax_rows_with_sink_batch ----------
// Gộp append sink + softmax cho từng (b,h).
// sink: [n_heads] của layer l (giống bạn đang dùng)

__global__ void k_softmax_rows_with_sink_batch(float* __restrict__ attB,
                                               const float* __restrict__ sink,
                                               const int*   __restrict__ positions,
                                               int n_heads, int row_stride,
                                               int B)
{
  const int h = blockIdx.x;  // head
  const int b = blockIdx.y;  // batch
  if (h >= n_heads || b >= B) return;

  const int tid   = threadIdx.x;
  const int lane  = tid & (WARP_SIZE-1);
  const int warp  = tid / WARP_SIZE;
  const int nwarps= blockDim.x / WARP_SIZE;

  float* row = attB + ((size_t)b * (size_t)n_heads + (size_t)h) * (size_t)row_stride;
  const int pos = positions[b];

  // row_len_core = pos+1 (đã có token 0..pos)
  const int row_len_core = pos + 1;
  const int sink_col     = row_len_core;      // sẽ đặt ở cột pos+1
  const int total_len    = row_len_core + 1;  // + sink

  // ---- Step 1: reduce max (bao gồm cả sink[h]) ----
  float local_max = -INFINITY;
  for (int i = tid; i < row_len_core; i += blockDim.x) local_max = fmaxf(local_max, row[i]);
  float sink_val = sink[h];
  local_max = fmaxf(local_max, sink_val);

  float wmax = warp_reduce_max(local_max);
  __shared__ float warp_buf[32];              // đủ cho <= 32 warps
  if (lane == 0) warp_buf[warp] = wmax;
  __syncthreads();

  float mx = -INFINITY;
  if (warp == 0) {
    float v = (lane < nwarps) ? warp_buf[lane] : -INFINITY;
    float g = warp_reduce_max(v);
    if (lane == 0) warp_buf[0] = g;
  }
  __syncthreads();
  mx = warp_buf[0];

  // ---- Step 2: exp & reduce sum (bao gồm sink) ----
  float local_sum = 0.f;
  for (int i = tid; i < row_len_core; i += blockDim.x) {
    float e = __expf(row[i] - mx);
    row[i]  = e;
    local_sum += e;
  }
  float e_sink = __expf(sink_val - mx);
  if (tid == 0) row[sink_col] = e_sink;   // ghi tạm exp(sink)
  local_sum += (tid == 0 ? e_sink : 0.f);

  float wsum = warp_reduce_sum(local_sum);
  if (lane == 0) warp_buf[warp] = wsum;
  __syncthreads();

  float sum = 0.f;
  if (warp == 0) {
    float v = (lane < nwarps) ? warp_buf[lane] : 0.f;
    float g = warp_reduce_sum(v);
    if (lane == 0) warp_buf[0] = g;
  }
  __syncthreads();
  sum = warp_buf[0];

  // ---- Step 3: normalize (bao gồm sink col) ----
  for (int i = tid; i < row_len_core; i += blockDim.x) {
    row[i] = row[i] / sum;
  }
  if (tid == 0) row[sink_col] = row[sink_col] / sum;

  // Kết quả: hàng có chiều dài hợp lệ = total_len = pos[b]+2.
}

// ===================== Launchers =====================

// softmax + sink (batch)
static inline void softmax_rows_with_sink_gpu_batch(float* attB,
                                                    const float* sink_h, // w->attn_sinks + l*n_heads
                                                    const int* d_positions,
                                                    int n_heads, int B,
                                                    int row_stride,
                                                    hipStream_t stream = 0)
{
  const int BS = 256; // up to ~2048 tokens ổn
  dim3 block(BS);
  dim3 grid(n_heads, B, 1);
  hipLaunchKernelGGL(k_softmax_rows_with_sink_batch, grid, block, 0, stream,
                     attB, sink_h, d_positions, n_heads, row_stride, B);
}

// ==== 1) Kernel: mỗi block xử lý 1 hàng; row_len cố định cho mọi hàng ====
__global__ void k_softmax_rows_batched_constlen(float *rows,
                                                int row_len, int row_stride,
                                                int n_rows) {
    const int rid = blockIdx.x;                   // row id
    if (rid >= n_rows) return;
    float *row = rows + (size_t)rid * (size_t)row_stride;

    const int tid   = threadIdx.x;
    const int lane  = tid & (WARP_SIZE - 1);
    const int warp  = tid / WARP_SIZE;
    const int nwarps= (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ float redbuf[32];                  // đủ cho ≤ 32 warps (<= 2048 thr @64)

    // Step 1: tìm max
    float local_max = -INFINITY;
    for (int i = tid; i < row_len; i += blockDim.x)
        local_max = fmaxf(local_max, row[i]);
    float wmax = warp_reduce_max(local_max);
    if (lane == 0) redbuf[warp] = wmax;
    __syncthreads();

    float mx = -INFINITY;
    if (warp == 0) {
        float v = (lane < nwarps) ? redbuf[lane] : -INFINITY;
        float g = warp_reduce_max(v);
        if (lane == 0) redbuf[0] = g;
    }
    __syncthreads();
    mx = redbuf[0];

    // Step 2: exp & sum
    float local_sum = 0.f;
    for (int i = tid; i < row_len; i += blockDim.x) {
        float e = __expf(row[i] - mx);
        row[i]  = e;
        local_sum += e;
    }
    float wsum = warp_reduce_sum(local_sum);
    if (lane == 0) redbuf[warp] = wsum;
    __syncthreads();

    float sum = 0.f;
    if (warp == 0) {
        float v = (lane < nwarps) ? redbuf[lane] : 0.f;
        float g = warp_reduce_sum(v);
        if (lane == 0) redbuf[0] = g;
    }
    __syncthreads();
    sum = redbuf[0];

    // Step 3: normalize
    for (int i = tid; i < row_len; i += blockDim.x)
        row[i] = row[i] / sum;
}

// ===================== Launchers =====================
static inline void softmax_rows_gpu_batch_constlen(float *rows,
                                                   int n_rows,
                                                   int row_len,
                                                   int row_stride,
                                                   hipStream_t stream=0) {
    const int BS = 256; // bội số của WARP_SIZE
    dim3 grid(n_rows), block(BS);
    hipLaunchKernelGGL(k_softmax_rows_batched_constlen, grid, block, 0, stream,
                       rows, row_len, row_stride, n_rows);
}
