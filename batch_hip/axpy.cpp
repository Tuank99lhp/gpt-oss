// axpy_batched.cu
// Batched AXPY for HIP (no external libs).
// X[b, i] += alpha * Y[b, i]   (or alpha[b] per-sample)
// Layout expected: contiguous row-major per sample: [B, N]

// ========================== batched (alpha chung) ==========================
// X[b*N + i] += alpha * Y[b*N + i]
__global__ void k_axpy_batch(float * __restrict__ X,
                             const float * __restrict__ Y,
                             float alpha, int N, int B) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // along N
  int b = blockIdx.y;                             // batch index
  if (b >= B || i >= N) return;

  size_t off = (size_t)b * (size_t)N + (size_t)i;
  X[off] += alpha * Y[off];
}
static inline void axpy_gpu_batch(float *X, const float *Y,
                                  float alpha, int N, int B,
                                  hipStream_t stream = 0) {
  const int BSx = 256;
  dim3 block(BSx, 1, 1);
  dim3 grid((N + BSx - 1) / BSx, B, 1);
  hipLaunchKernelGGL(k_axpy_batch, grid, block, 0, stream, X, Y, alpha, N, B);
}
