// add_bias_batched.cu
// Batch bias add for HIP (no external libs)
// Layout (row-major):
//   Y: [B, N] contiguous along N
//   Bias broadcast: b: [N]
//   Bias per-sample: b: [B, N]
//
// Build: hipcc -O3 -std=c++17 -o add_bias_batched add_bias_batched.cu

// ============================== Kernels ===============================

// --- 1) Broadcast bias: Y[b, i] += b[i] ---
template<bool VEC4=true>
__global__ void k_add_bias_batch_broadcast(float* __restrict__ Y,  // [B,N]
                                           const float* __restrict__ b,  // [N]
                                           int B, int N) {
  // Each block works on one row 'bidx' (via grid.y), and strided columns on grid.x
  const int bidx = blockIdx.y;
  if (bidx >= B) return;

  float*       __restrict__ yrow = Y + (size_t)bidx * (size_t)N;
  const float* __restrict__ brow = b;

  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x  * blockDim.x;

  const bool aligned16 = (((uintptr_t)yrow & 0xF) == 0) && (((uintptr_t)brow & 0xF) == 0);

  if (VEC4 && aligned16) {
    // Vectorized path: operate 4 floats at a time. Guard the tail in-loop.
    for (int i = tid * 4; i < N; i += stride * 4) {
      if (i + 3 < N) {
        Float4*       y4 = reinterpret_cast<Float4*>(yrow + i);
        const Float4* b4 = reinterpret_cast<const Float4*>(brow + i);
        Float4 a = *y4;
        Float4 c = *b4;
        a.x += c.x; a.y += c.y; a.z += c.z; a.w += c.w;
        *y4 = a;
      } else {
        // scalar tail for this thread's last partial vector
        for (int j = i; j < N; ++j) yrow[j] += brow[j];
      }
    }
  } else {
    // Scalar fallback
    for (int i = tid; i < N; i += stride) yrow[i] += brow[i];
  }
}

// --- 2) Per-sample bias: Y[b, i] += b[b, i] ---
template<bool VEC4=true>
__global__ void k_add_bias_batch_per_sample(float* __restrict__ Y,      // [B,N]
                                            const float* __restrict__ BIAS, // [B,N]
                                            int B, int N) {
  const int bidx = blockIdx.y;
  if (bidx >= B) return;

  float*       __restrict__ yrow = Y     + (size_t)bidx * (size_t)N;
  const float* __restrict__ brow = BIAS  + (size_t)bidx * (size_t)N;

  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x  * blockDim.x;

  const bool aligned16 = (((uintptr_t)yrow & 0xF) == 0) &&
                         (((uintptr_t)brow & 0xF) == 0);

  if (VEC4 && aligned16) {
    for (int i = tid * 4; i < N; i += stride * 4) {
      if (i + 3 < N) {
        Float4*       y4 = reinterpret_cast<Float4*>(yrow + i);
        const Float4* b4 = reinterpret_cast<const Float4*>(brow + i);
        Float4 a = *y4;
        Float4 c = *b4;
        a.x += c.x; a.y += c.y; a.z += c.z; a.w += c.w;
        *y4 = a;
      } else {
        for (int j = i; j < N; ++j) yrow[j] += brow[j];
      }
    }
  } else {
    for (int i = tid; i < N; i += stride) yrow[i] += brow[i];
  }
}

// ============================== Launchers ===============================

// Heuristic: grid.x big enough to saturate memory BW; grid.y = B.
static inline void add_bias_gpu_batch_broadcast(float* Y, const float* b,
                                                int B, int N,
                                                hipStream_t stream = 0) {
  const int BS = 256;
  // use ~ceil(N / (BS*vec)) for grid.x; keep at least 1
  const int vec = 4; // we attempt float4 when aligned
  int gx = (N + (BS*vec - 1)) / (BS*vec);
  if (gx < 1) gx = 1;

  dim3 block(BS, 1, 1);
  dim3 grid(gx, B, 1);

  hipLaunchKernelGGL((k_add_bias_batch_broadcast<true>), grid, block, 0, stream,
                     Y, b, B, N);
}

// Per-sample bias launcher
static inline void add_bias_gpu_batch_per_sample(float* Y, const float* BIAS,
                                                 int B, int N,
                                                 hipStream_t stream = 0) {
  const int BS = 256;
  const int vec = 4;
  int gx = (N + (BS*vec - 1)) / (BS*vec);
  if (gx < 1) gx = 1;

  dim3 block(BS, 1, 1);
  dim3 grid(gx, B, 1);

  hipLaunchKernelGGL((k_add_bias_batch_per_sample<true>), grid, block, 0, stream,
                     Y, BIAS, B, N);
}