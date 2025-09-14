// RMSNorm batched: out[b, i] = w[i] * x[b, i] / sqrt(mean_j x[b, j]^2 + eps)
__global__ void k_rmsnorm_batch(float * __restrict__ out,     // [B, H]
                                const float * __restrict__ x, // [B, H]
                                const float * __restrict__ w, // [H] (shared cho mọi b)
                                int H, float eps) {
  extern __shared__ double ssum[]; // size = blockDim.x
  const int b   = blockIdx.x;      // 1 block ↔ 1 sample
  const int tid = threadIdx.x;

  // 1) accumulate sum of squares (double) theo stride
  double acc = 0.0;
  const float* xb = x + (size_t)b * (size_t)H;
  for (int i = tid; i < H; i += blockDim.x) {
    float v = xb[i];
    acc += (double)v * (double)v;
  }
  ssum[tid] = acc;
  __syncthreads();

  // 2) block-wide reduction
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) ssum[tid] += ssum[tid + s];
    __syncthreads();
  }

  // 3) tính inv_rms và ghi kết quả (theo stride)
  const float inv_rms = rsqrtf((float)(ssum[0] / (double)H + (double)eps));
  float* ob = out + (size_t)b * (size_t)H;
  for (int i = tid; i < H; i += blockDim.x) {
    ob[i] = w[i] * (xb[i] * inv_rms);
  }
}

// Wrapper tiện dụng
static inline void rmsnorm_batch_gpu(float *out, const float *x,
                                     const float *weight,
                                     int B, int H, float eps = 1e-5f,
                                     hipStream_t stream = 0) {
  // chọn block size hợp lý; stride loop sẽ cover H
  const int BS = (H >= 1024) ? 1024 : (H >= 512 ? 512 : 256);
  size_t shmem = (size_t)BS * sizeof(double);
  hipLaunchKernelGGL(k_rmsnorm_batch, dim3(B), dim3(BS), shmem, stream,
                     out, x, weight, H, eps);
}
