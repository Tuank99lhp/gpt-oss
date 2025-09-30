__global__ void k_moe_assign_from_topk(const int*   __restrict__ topk_idx, // [B*K]
                                       const float* __restrict__ topk_val, // [B*K]
                                       const float* __restrict__ batch_t,  // [B,H]
                                       int H, int B, int K, int E,
                                       int*        __restrict__ counts,        // [E]
                                       int*        __restrict__ idx_in_batch,  // [E*B]
                                       float**     __restrict__ in_ptrs,       // [E*B] (mảng con trỏ)
                                       float*      __restrict__ wexps)         // [E*B]
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int NK  = B * K;
  if (tid >= NK) return;

  int b = tid / K;
  int k = tid % K;

  int e = topk_idx[tid];
  float w = topk_val[tid];
  if ((unsigned)e >= (unsigned)E) return;

  // vị trí mới trong batch của expert e
  int pos = atomicAdd(&counts[e], 1);

  if (pos < B) {
    idx_in_batch[e * B + b] = pos;
    // trỏ tới dòng b trong batch_t
    // LƯU Ý: đây là con trỏ device -> device; HIP hỗ trợ viết pointer vào memory device.
    in_ptrs[e * B + pos] = const_cast<float*>(batch_t + b * H);
    wexps  [e * B + pos] = w;
  }
}

void moe_assign_from_topk(
    const int*   __restrict__ topk_idx, // [B*K]
    const float* __restrict__ topk_val, // [B*K]
    const float* __restrict__ batch_t,  // [B,H]
    int H, int B, int K, int E,
    int*        __restrict__ counts,        // [E]
    int*        __restrict__ idx_in_batch,  // [E*B]
    float**     __restrict__ in_ptrs,       // [E*B] (mảng con trỏ)
    float*      __restrict__ wexps,         // [E*B]
    hipStream_t stream = 0)
{
  if (H <= 0 || B <= 0 || K <= 0 || E <= 0) return;

  int NK = B * K;
  int block = 256;
  int grid  = ceil_div(NK, block);

  hipLaunchKernelGGL(
      k_moe_assign_from_topk,
      grid, block,
      /*sharedMemBytes=*/0, stream,
      topk_idx, topk_val, batch_t,
      H, B, K, E,
      counts, idx_in_batch, in_ptrs, wexps);
}

__global__ void k_moe_aggregate_topk(const int*   __restrict__ topk_idx, // [B*K]
                                     const float* __restrict__ topk_val, // [B*K]
                                     const int*   __restrict__ idx_in_batch, // [E*B]
                                     float*       __restrict__ out_bases,   // [E] -> each [Be,H]
                                     float*       __restrict__ batch_x,      // [B,H] (cộng dồn)
                                     int H, int B, int K, int E)
{
  int b = blockIdx.y;               // 0..B-1
  int h = blockIdx.x * blockDim.x + threadIdx.x; // 0..H-1
  if (b >= B || h >= H) return;

  float acc = 0.f;
  int baseBK = b * K;

  #pragma unroll
  for (int kk = 0; kk < 8; ++kk) { // K <= 8 (theo code của bạn), nếu lớn hơn, bỏ unroll
    if (kk >= K) break;
    int e = topk_idx[baseBK + kk];
    if ((unsigned)e >= (unsigned)E) continue;
    int idx = idx_in_batch[e * B + b];  // hàng trong output expert e ứng với token b
    if (idx < 0) continue;              // token không thuộc expert này (an toàn)
    const float* out_e = out_bases + e * B * H;  // base pointer expert e
    acc += topk_val[baseBK + kk] * out_e[idx * H + h];
  }

  // cộng dồn vào batch_x
  batch_x[b * H + h] += acc;
}

void moe_aggregate_topk(
    const int*   __restrict__ topk_idx, // [B*K]
    const float* __restrict__ topk_val, // [B*K]
    const int*   __restrict__ idx_in_batch, // [E*B]
    float*       __restrict__ out_bases,   // [E] -> each [Be,H]
    float*       __restrict__ batch_x,      // [B,H] (cộng dồn)
    int H, int B, int K, int E,
    hipStream_t stream = 0)
{
  if (H <= 0 || B <= 0 || K <= 0 || E <= 0) return;

  int block = 256;
  int gridx = ceil_div(H, block);
  int gridy = B;
  dim3 grid(gridx, gridy);

  hipLaunchKernelGGL(
      k_moe_aggregate_topk,
      grid, block,
      /*sharedMemBytes=*/0, stream,
      topk_idx, topk_val, idx_in_batch, out_bases, batch_x,
      H, B, K, E);
}

__global__ void k_moe_assign_from_topk_bf16(const int*   __restrict__ topk_idx, // [B*K]
                                       const float* __restrict__ topk_val, // [B*K]
                                       const hip_bfloat16* __restrict__ batch_t,  // [B,H]
                                       int H, int B, int K, int E,
                                       int*        __restrict__ counts,        // [E]
                                       int*        __restrict__ idx_in_batch,  // [E*B]
                                       hip_bfloat16**     __restrict__ in_ptrs,       // [E*B] (mảng con trỏ)
                                       float*      __restrict__ wexps)         // [E*B]
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int NK  = B * K;
  if (tid >= NK) return;

  int b = tid / K;
  int k = tid % K;

  int e = topk_idx[tid];
  float w = topk_val[tid];
  if ((unsigned)e >= (unsigned)E) return;

  // vị trí mới trong batch của expert e
  int pos = atomicAdd(&counts[e], 1);

  if (pos < B) {
    idx_in_batch[e * B + b] = pos;
    // trỏ tới dòng b trong batch_t
    // LƯU Ý: đây là con trỏ device -> device; HIP hỗ trợ viết pointer vào memory device.
    in_ptrs[e * B + pos] = const_cast<hip_bfloat16*>(batch_t + b * H);
    wexps  [e * B + pos] = w;
  }
}

void moe_assign_from_topk_bf16(
    const int*   __restrict__ topk_idx, // [B*K]
    const float* __restrict__ topk_val, // [B*K]
    const hip_bfloat16* __restrict__ batch_t,  // [B,H]
    int H, int B, int K, int E,
    int*        __restrict__ counts,        // [E]
    int*        __restrict__ idx_in_batch,  // [E*B]
    hip_bfloat16**     __restrict__ in_ptrs,       // [E*B] (mảng con trỏ)
    float*      __restrict__ wexps,         // [E*B]
    hipStream_t stream = 0)
{
  if (H <= 0 || B <= 0 || K <= 0 || E <= 0) return;

  int NK = B * K;
  int block = 256;
  int grid  = ceil_div(NK, block);

  hipLaunchKernelGGL(
      k_moe_assign_from_topk_bf16,
      grid, block,
      /*sharedMemBytes=*/0, stream,
      topk_idx, topk_val, batch_t,
      H, B, K, E,
      counts, idx_in_batch, in_ptrs, wexps);
}