// add_bias_batched.cu
// Batch bias add for HIP (no external libs)
// Layout (row-major):
//   Y: [B, N] contiguous along N
//   Bias broadcast: b: [N]
//   Bias per-sample: b: [B, N]
//
// Build: hipcc -O3 -std=c++17 -o add_bias_batched add_bias_batched.cu

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef HIP_CHECK
#define HIP_CHECK(cmd) do { \
  hipError_t _e = (cmd);    \
  if (_e != hipSuccess) {   \
    fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
            (int)_e, hipGetErrorString(_e), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)
#endif

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

// ============================== (Optional) Test =============================
// #define TEST_ADD_BIAS_BATCH
#ifdef TEST_ADD_BIAS_BATCH
#include <vector>
#include <math.h>
int main() {
  int B=3, N=1000;
  std::vector<float> hY(B*N, 1.0f), hB(N, 0.5f), hYP(B*N, 1.0f), hBP(B*N, 0.0f);
  for (int b=0;b<B;b++) for (int i=0;i<N;i++) hBP[b*N+i] = 0.1f*b + 0.01f*i;

  float *dY, *dB, *dYP, *dBP;
  HIP_CHECK(hipMalloc(&dY,  B*N*sizeof(float)));
  HIP_CHECK(hipMalloc(&dB,  N*sizeof(float)));
  HIP_CHECK(hipMalloc(&dYP, B*N*sizeof(float)));
  HIP_CHECK(hipMalloc(&dBP, B*N*sizeof(float)));
  HIP_CHECK(hipMemcpy(dY,  hY.data(),  B*N*sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB,  hB.data(),  N*sizeof(float),   hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dYP, hYP.data(), B*N*sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dBP, hBP.data(), B*N*sizeof(float), hipMemcpyHostToDevice));

  add_bias_gpu_batch_broadcast(dY, dB, B, N);
  add_bias_gpu_batch_per_sample(dYP, dBP, B, N);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(hY.data(),  dY,  B*N*sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(hYP.data(), dYP, B*N*sizeof(float), hipMemcpyDeviceToHost));

  // quick checks
  double max_abs0=0.0, max_abs1=0.0;
  for (int b=0;b<B;b++) {
    for (int i=0;i<N;i++) {
      double ref0 = 1.0 + 0.5;
      double ref1 = 1.0 + (0.1*b + 0.01*i);
      max_abs0 = fmax(max_abs0, fabs(hY[b*N+i]-ref0));
      max_abs1 = fmax(max_abs1, fabs(hYP[b*N+i]-ref1));
    }
  }
  printf("broadcast max_abs=%.3g, per_sample max_abs=%.3g\n",
         max_abs0, max_abs1);

  hipFree(dY); hipFree(dB); hipFree(dYP); hipFree(dBP);
  return 0;
}
#endif
