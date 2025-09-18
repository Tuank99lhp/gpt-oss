// ===================== Batched Attention Kernels =====================

// ---------- k_softmax_rows_with_sink_batch ----------
// Gộp append sink + softmax cho từng (b,h).
// sink: [n_heads] của layer l (giống bạn đang dùng)
__device__ __forceinline__ float warp_max(float v){
  for (int off = WARP_SIZE>>1; off>0; off>>=1) v = fmaxf(v, __shfl_down(v, off, WARP_SIZE));
  return v;
}
__device__ __forceinline__ float warp_sum(float v){
  for (int off = WARP_SIZE>>1; off>0; off>>=1) v += __shfl_down(v, off, WARP_SIZE);
  return v;
}

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

  float wmax = warp_max(local_max);
  __shared__ float warp_buf[32];              // đủ cho <= 32 warps
  if (lane == 0) warp_buf[warp] = wmax;
  __syncthreads();

  float mx = -INFINITY;
  if (warp == 0) {
    float v = (lane < nwarps) ? warp_buf[lane] : -INFINITY;
    float g = warp_max(v);
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

  float wsum = warp_sum(local_sum);
  if (lane == 0) warp_buf[warp] = wsum;
  __syncthreads();

  float sum = 0.f;
  if (warp == 0) {
    float v = (lane < nwarps) ? warp_buf[lane] : 0.f;
    float g = warp_sum(v);
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

// ==== 2) Kernel: mỗi block xử lý 1 hàng; row_len thay đổi theo b ====
__global__ void k_softmax_rows_batched_varlen(float *rows,
                                              const int *positions, // [B]
                                              int n_heads,          // per-batch rows
                                              int row_stride,
                                              int B,
                                              int row_extra) {      // 0 (không sink) / 1 (có sink)
    const int rid = blockIdx.x;                     // 0..B*n_heads-1
    const int b = rid / n_heads;
    const int h = rid - b * n_heads;
    if (b >= B) return;

    float *row = rows + ((size_t)b * (size_t)n_heads + (size_t)h) * (size_t)row_stride;
    const int row_len = positions[b] + row_extra;   // ví dụ pos+2 nếu trước đó đã append sink; ở đây thường dùng pos+2 khi đã đặt sink vào cột pos+1

    const int tid   = threadIdx.x;
    const int lane  = tid & (WARP_SIZE - 1);
    const int warp  = tid / WARP_SIZE;
    const int nwarps= (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ float redbuf[32];

    // Step 1: max
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

static inline void softmax_rows_gpu_batch_varlen(float *rows,
                                                 const int *d_positions, // [B] device
                                                 int B,
                                                 int n_heads,
                                                 int row_extra,          // 1 nếu đã thêm sink vào cột pos+1 (tức row_len = pos+2), 0 nếu không
                                                 int row_stride,
                                                 hipStream_t stream=0) {
    const int BS = 256;
    const int n_rows = B * n_heads;
    dim3 grid(n_rows), block(BS);
    hipLaunchKernelGGL(k_softmax_rows_batched_varlen, grid, block, 0, stream,
                       rows, d_positions, n_heads, row_stride, B, row_extra);
}

// warp reduce sum
__inline__ __device__ float warp_reduce_sum_softmax(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down(v, offset);
    return v;
}

__global__ void k_softmax_rows(float *att, int row_len, int row_stride) {
    int head = blockIdx.x;
    float *row = att + (size_t)head * row_stride;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    // Step 1: find max
    float local_max = -INFINITY;
    for (int i = tid; i < row_len; i += blockDim.x) {
        local_max = fmaxf(local_max, row[i]);
    }
    float warp_max = warp_reduce_max(local_max);

    __shared__ float block_max[8]; // supports up to 8 warps (256 threads)
    if (lane == 0) block_max[warp] = warp_max;
    __syncthreads();
    float mx = -INFINITY;
    if (warp == 0) {
        float val = (tid < (blockDim.x>>5)) ? block_max[lane] : -INFINITY;
        float wmax = warp_reduce_max(val);
        if (lane == 0) block_max[0] = wmax;
    }
    __syncthreads();
    mx = block_max[0];

    // Step 2: exp and local sum
    float local_sum = 0.f;
    for (int i = tid; i < row_len; i += blockDim.x) {
        float v = __expf(row[i] - mx);
        row[i] = v;
        local_sum += v;
    }
    float warp_sum = warp_reduce_sum_softmax(local_sum);
    if (lane == 0) block_max[warp] = warp_sum;
    __syncthreads();
    float sum = 0.f;
    if (warp == 0) {
        float val = (tid < (blockDim.x>>5)) ? block_max[lane] : 0.f;
        float wsum = warp_reduce_sum_softmax(val);
        if (lane == 0) block_max[0] = wsum;
    }
    __syncthreads();
    sum = block_max[0];

    // Step 3: normalize
    for (int i = tid; i < row_len; i += blockDim.x) {
        row[i] = row[i] / sum;
    }
}

static void softmax_rows_gpu(float *att, int n_heads, int row_len, int row_stride) {
    int BS = 256; // good for row_len up to 2048
    dim3 grid(n_heads);
    hipLaunchKernelGGL(k_softmax_rows, grid, dim3(BS), 0, 0, att, row_len, row_stride);
}

