// gemv_batched_universal.cu
// Single-kernel batched GEMV for HIP (no hipBLAS).
// Y[b, m] = sum_k W[m, k] * X[b, k]
// One kernel, works for any B (MAX_BATCH_SIZE), M, K. Vectorized global loads
// only when safe; shared-memory stores are scalar to avoid alignment pitfalls.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

// ---- Tunable tiles (safe defaults)
#ifndef TILE_B
#define TILE_B  16   // samples per block (tile size along B)
#endif
#ifndef TILE_M
#define TILE_M  64   // output rows per block (tile size along M)
#endif
#ifndef TILE_K
#define TILE_K  64   // reduction tile along K (keep multiple of 8/16 for perf)
#endif

// 16B vector type for global loads
// struct __align__(16) Float4 { float x, y, z, w; };

template <typename T>
__host__ __device__ __forceinline__ T dmin(T a, T b) { return a < b ? a : b; }

// ==================== Single universal kernel ====================
//
// Grid : grid = (ceil(M/TM), ceil(B/TB), 1)
// Block: block = (TM, TB, 1) → mỗi thread tính 1 ô Y trong tile.
// LDS  : Xsh[TB, TK], Wsh[TK, TM]; W tile nạp và lưu dạng chuyển vị (k, m).
//
template<int TB=TILE_B, int TM=TILE_M, int TK=TILE_K, bool TRY_VEC4=true>
__global__ void k_gemv_batch_unified(const float* __restrict__ W,   // [M,K]
                                     const float* __restrict__ X,   // [B,K]
                                     float*       __restrict__ Y,   // [B,M]
                                     int K, int M, int B) {
  // Align shared to 16B; but we will only write scalar to avoid alignment traps.
  __shared__ __align__(16) float Xsh[TB][TK];
  __shared__ __align__(16) float Wsh[TK][TM];

  const int ty = threadIdx.y;   // 0..TB-1  (row in Xsh)
  const int tx = threadIdx.x;   // 0..TM-1  (col in Wsh / output col in tile)

  const int b0 = blockIdx.y * TB;   // batch start for this block
  const int m0 = blockIdx.x * TM;   // row start   for this block

  const int b  = b0 + ty;           // global batch index
  const int m  = m0 + tx;           // global row index

  float acc = 0.0f;

  // Vectorization safety (global loads only):
  const bool X_base16 = (((uintptr_t)X & 0xF) == 0);
  const bool W_base16 = (((uintptr_t)W & 0xF) == 0);
  const bool stride16 = ((K & 3) == 0);    // row stride (K floats) multiple of 16B

  // Cooperative loading across the CTA:
  const int threadsPerBlock = blockDim.x * blockDim.y;
  const int tidLinear = ty * blockDim.x + tx;

  for (int k0 = 0; k0 < K; k0 += TK) {
    const int k_lim = dmin(TK, K - k0);

    // SAFE to read float4 from global if base aligned, stride aligned, k0 aligned:
    const bool vec4_ok = TRY_VEC4 && stride16 && ((k0 & 3) == 0) && (k_lim >= 4);

    // -------- Load X tile: [TB, k_lim] into Xsh --------
    if (vec4_ok && X_base16) {
      // main vectorized part
      const int cols4 = (k_lim >> 2);
      const int total4 = TB * cols4;
      for (int idx4 = tidLinear; idx4 < total4; idx4 += threadsPerBlock) {
        int tb  = idx4 / cols4;
        int tk4 = idx4 - tb * cols4;
        int gb  = b0 + tb;
        int gk  = k0 + (tk4 << 2);
        Float4 v = {0,0,0,0};
        if (gb < B) {
          const Float4* px = reinterpret_cast<const Float4*>(
              X + (size_t)gb * (size_t)K + (size_t)gk);
          v = *px;  // global aligned float4 load
        }
        // store to shared in SCALAR (avoid alignment constraints in LDS)
        if ( (tk4<<2) + 0 < k_lim ) Xsh[tb][(tk4<<2) + 0] = v.x;
        if ( (tk4<<2) + 1 < k_lim ) Xsh[tb][(tk4<<2) + 1] = v.y;
        if ( (tk4<<2) + 2 < k_lim ) Xsh[tb][(tk4<<2) + 2] = v.z;
        if ( (tk4<<2) + 3 < k_lim ) Xsh[tb][(tk4<<2) + 3] = v.w;
      }
      // tail part (if any)
      const int vec_end = (cols4 << 2);
      const int tail = k_lim - vec_end;
      if (tail) {
        const int span = TB * tail;
        for (int idx = tidLinear; idx < span; idx += threadsPerBlock) {
          int tb = idx / tail;
          int tk = idx - tb * tail;
          int gb = b0 + tb;
          int gk = k0 + vec_end + tk;
          Xsh[tb][vec_end + tk] = (gb < B) ? X[(size_t)gb * (size_t)K + (size_t)gk] : 0.0f;
        }
      }
    } else {
      // Scalar cooperative load
      const int span = TB * k_lim;
      for (int idx = tidLinear; idx < span; idx += threadsPerBlock) {
        int tb = idx / k_lim;
        int tk = idx - tb * k_lim;
        int gb = b0 + tb;
        int gk = k0 + tk;
        Xsh[tb][tk] = (gb < B) ? X[(size_t)gb * (size_t)K + (size_t)gk] : 0.0f;
      }
    }

    // -------- Load W tile: [k_lim, TM] into Wsh (transposed: k→row, m→col) --------
    if (vec4_ok && W_base16) {
      const int cols4 = (k_lim >> 2);
      const int total4 = TM * cols4;
      for (int idx4 = tidLinear; idx4 < total4; idx4 += threadsPerBlock) {
        int tm  = idx4 / cols4;      // 0..TM-1
        int tk4 = idx4 - tm * cols4; // 0..cols4-1
        int gm  = m0 + tm;
        int gk  = k0 + (tk4 << 2);
        Float4 w4 = {0,0,0,0};
        if (gm < M) {
          const Float4* pw = reinterpret_cast<const Float4*>(
              W + (size_t)gm * (size_t)K + (size_t)gk);
          w4 = *pw;  // global aligned float4 load
        }
        // store to shared in SCALAR (avoid LDS alignment issues)
        if ( (tk4<<2) + 0 < k_lim ) Wsh[(tk4<<2) + 0][tm] = w4.x;
        if ( (tk4<<2) + 1 < k_lim ) Wsh[(tk4<<2) + 1][tm] = w4.y;
        if ( (tk4<<2) + 2 < k_lim ) Wsh[(tk4<<2) + 2][tm] = w4.z;
        if ( (tk4<<2) + 3 < k_lim ) Wsh[(tk4<<2) + 3][tm] = w4.w;
      }
      // tail
      const int vec_end = (cols4 << 2);
      const int tail = k_lim - vec_end;
      if (tail) {
        const int span = TM * tail;
        for (int idx = tidLinear; idx < span; idx += threadsPerBlock) {
          int tm = idx / tail;
          int tk = idx - tm * tail;
          int gm = m0 + tm;
          int gk = k0 + vec_end + tk;
          float w = (gm < M) ? W[(size_t)gm * (size_t)K + (size_t)gk] : 0.0f;
          Wsh[vec_end + tk][tm] = w;
        }
      }
    } else {
      // Scalar cooperative load
      const int span = TM * k_lim;
      for (int idx = tidLinear; idx < span; idx += threadsPerBlock) {
        int tm = idx / k_lim;
        int tk = idx - tm * k_lim;
        int gm = m0 + tm;
        int gk = k0 + tk;
        float w = (gm < M) ? W[(size_t)gm * (size_t)K + (size_t)gk] : 0.0f;
        Wsh[tk][tm] = w;
      }
    }

    __syncthreads();

    // -------- Compute MAC on tile --------
    if (b < B && m < M) {
      #pragma unroll
      for (int tk = 0; tk < k_lim; ++tk) {
        acc = fmaf(Xsh[ty][tk], Wsh[tk][tx], acc);
      }
    }

    __syncthreads(); // reuse LDS safely for next K-tile
  }

  // -------- Store output --------
  if (b < B && m < M) {
    Y[(size_t)b * (size_t)M + (size_t)m] = acc;
  }
}

// ==================== Single universal launcher ====================
static inline void gemv_gpu_batch(float* Y, const float* X, const float* W,
                                  int K, int M, int B,
                                  hipStream_t stream = 0) {
  dim3 block(TILE_M, TILE_B, 1);  // (TM, TB)
  dim3 grid((M + TILE_M - 1) / TILE_M,
            (B + TILE_B - 1) / TILE_B,
            1);
  hipLaunchKernelGGL((k_gemv_batch_unified<TILE_B, TILE_M, TILE_K, /*TRY_VEC4=*/true>),
                     grid, block, 0, stream, W, X, Y, K, M, B);
}

// ======================== (Optional) tiny test =====================
// #define GEMV_BATCH_TEST
#ifdef GEMV_BATCH_TEST
#include <vector>
int main() {
  // Try tricky sizes (K not divisible by 4) to stress non-vec path, and B large
  int B=17, M=123, K=257;
  std::vector<float> hX(B*K), hW(M*K), hY(B*M), hRef(B*M);

  for (int b=0;b<B;b++) for (int k=0;k<K;k++)
    hX[b*K+k] = 0.1f*(b+1) + 0.001f*k;
  for (int m=0;m<M;m++) for (int k=0;k<K;k++)
    hW[m*K+k] = 0.2f*(m+1) - 0.003f*k;

  float *dX,*dW,*dY;
  HIP_CHECK(hipMalloc(&dX, B*K*sizeof(float)));
  HIP_CHECK(hipMalloc(&dW, M*K*sizeof(float)));
  HIP_CHECK(hipMalloc(&dY, B*M*sizeof(float)));
  HIP_CHECK(hipMemcpy(dX, hX.data(), B*K*sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dW, hW.data(), M*K*sizeof(float), hipMemcpyHostToDevice));

  gemv_gpu_batch(dY, dX, dW, K, M, B);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(hY.data(), dY, B*M*sizeof(float), hipMemcpyDeviceToHost));

  // CPU ref
  for (int b=0;b<B;b++) for (int m=0;m<M;m++) {
    double acc = 0.0;
    for (int k=0;k<K;k++) acc += (double)hW[m*K+k]*(double)hX[b*K+k];
    hRef[b*M+m] = (float)acc;
  }
  double max_abs=0.0;
  for (int i=0;i<B*M;i++) max_abs = fmax(max_abs, fabs((double)hRef[i] - (double)hY[i]));
  printf("max_abs=%.6g  %s\n", max_abs, (max_abs<1e-4) ? "OK":"MISMATCH");

  hipFree(dX); hipFree(dW); hipFree(dY);
  return 0;
}
#endif
