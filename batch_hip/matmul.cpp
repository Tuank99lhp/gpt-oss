#ifndef BM
#define BM 32               // block tile on B dimension
#endif
#ifndef BN
#define BN 64               // block tile on M dimension
#endif
#ifndef BK
#define BK 64              // K step per LDS tile (must be multiple of 4)
#endif
#ifndef WAVES_M
#define WAVES_M 2           // waves along B (rows of Y)
#endif
#ifndef WAVES_N
#define WAVES_N 4           // waves along M (cols of Y)
#endif
static_assert(BM == 16*WAVES_M, "BM must be 16*WAVES_M");
static_assert(BN == 16*WAVES_N, "BN must be 16*WAVES_N");
static_assert(BK % 4 == 0, "BK must be multiple of 4 for 16x16x4 MFMA");

#ifndef APAD
#define APAD 1              // +1 padding column for LDS A-tile
#endif
#ifndef BPAD
#define BPAD 1              // +1 padding column for LDS B-tile
#endif

// MFMA accumulator type: vector of 4 float (matches builtin return type)
using v4f32 = float __attribute__((__vector_size__(4 * sizeof(float))));

// BF16 -> F32 fast convert (weight-friendly)
__device__ __forceinline__ float bf16_to_f32(hip_bfloat16 h) {
  uint16_t lo = *reinterpret_cast<const uint16_t*>(&h);
  uint32_t hi = (uint32_t)lo << 16;
  return __uint_as_float(hi);
}

// Convert 4×bf16 packed in one uint64 -> 4×f32
__device__ __forceinline__ void bf16x4_u64_to_f4(uint64_t u, float &x0, float &x1, float &x2, float &x3) {
  uint32_t lo = (uint32_t)(u & 0xFFFFFFFFull);
  uint32_t hi = (uint32_t)(u >> 32);
  uint32_t h0 = (lo & 0x0000FFFFu) << 16;
  uint32_t h1 = (lo & 0xFFFF0000u);
  uint32_t h2 = (hi & 0x0000FFFFu) << 16;
  uint32_t h3 = (hi & 0xFFFF0000u);
  x0 = __uint_as_float(h0);
  x1 = __uint_as_float(h1);
  x2 = __uint_as_float(h2);
  x3 = __uint_as_float(h3);
}

__global__ void k_gemm_bf16_opt_mfma_2x2waves(
    const float*         __restrict__ X,  // [B,K] f32
    const hip_bfloat16*  __restrict__ W,  // [K,M] bf16 (transposed already)
    float*               __restrict__ Y,  // [B,M] f32
    int K, int M, int B)
{
  // Wave-local lane coords for MFMA (16x16x4)
  const int tx = threadIdx.x; // 0..15
  const int ty = threadIdx.y; // 0..3
  const int wz = threadIdx.z; // 0..(WAVES_M*WAVES_N-1)

  // Wave indices inside block tile
  const int wave_m = wz / WAVES_N;  // 0..WAVES_M-1  (rows)
  const int wave_n = wz % WAVES_N;  // 0..WAVES_N-1  (cols)

  // Block origin in Y
  const int m0 = blockIdx.x * BN;   // col start
  const int b0 = blockIdx.y * BM;   // row start

  // Shared memory (double-buffer)
  __shared__ float sA[2][BM][BK + APAD];
  __shared__ float sB[2][BK][BN + BPAD];

  // Accumulator for this thread (4 partial sums)
  v4f32 acc = {0.f, 0.f, 0.f, 0.f};

  // Flattened thread id in block (0..255) to distribute loads
  const int tid = (wz * (blockDim.y * blockDim.x)) + (ty * blockDim.x + tx);

  // ---------- Lambda: cooperative load of one K-tile into LDS ----------
  auto load_k_tile_into_lds = [&](int buf, int k0) {
    // ---- Load X tile: [BM, BK] floats into sA[buf]
    // Vectorized by float4 when aligned; fallback scalar otherwise.
    // Number of float4 chunks in the tile:
    const int A_F4_W = BK / 4;                // groups of 4 along K
    const int A_TOT_F4 = BM * A_F4_W;         // total float4 to load
    for (int idx = tid; idx < A_TOT_F4; idx += blockDim.x * blockDim.y * blockDim.z) {
      int r  = idx / A_F4_W;                  // row in [0..BM)
      int c4 = idx % A_F4_W;                  // col-group in [0..BK/4)
      int b_row = b0 + r;
      int kk    = k0 + c4 * 4;

      float4 v = {0,0,0,0};
      if (b_row < B && (kk + 3) < K) {
        const float* gptr = X + (size_t)b_row * (size_t)K + (size_t)kk;
        // Vectorized if 16B aligned, else scalar gather
        if ((((uintptr_t)gptr) & 0xF) == 0) {
          v = *reinterpret_cast<const float4*>(gptr);
        } else {
          v.x = gptr[0];
          v.y = gptr[1];
          v.z = gptr[2];
          v.w = gptr[3];
        }
      }
      // Store to LDS with +APAD padding
      sA[buf][r][c4*4 + 0] = v.x;
      sA[buf][r][c4*4 + 1] = v.y;
      sA[buf][r][c4*4 + 2] = v.z;
      sA[buf][r][c4*4 + 3] = v.w;
    }

    // ---- Load W tile: [BK, BN] from W[kk, m] (BF16) -> float into sB[buf]
    // Vectorized by 4×bf16 (uint64)
    const int B_BF4_W = BN / 4;               // groups of 4 along M
    const int B_TOT_BF4 = BK * B_BF4_W;       // total bf16x4 packs to load
    for (int idx = tid; idx < B_TOT_BF4; idx += blockDim.x * blockDim.y * blockDim.z) {
      int r_k = idx / B_BF4_W;                // k in [0..BK)
      int c4  = idx % B_BF4_W;                // col-group in [0..BN/4)
      int kk  = k0 + r_k;                     // global k
      int mc  = m0 + c4 * 4;                  // global m start (4-wide)

      float f0=0,f1=0,f2=0,f3=0;
      if (kk < K) {
        if ((mc + 3) < M) {
          const uint64_t* gptr = reinterpret_cast<const uint64_t*>(W + (size_t)kk * (size_t)M + (size_t)mc);
          uint64_t u = *gptr; // contiguous across M (row-major [K,M])
          bf16x4_u64_to_f4(u, f0, f1, f2, f3);
        } else {
          // tail-safe (rare for your sizes)
          const hip_bfloat16* wp = W + (size_t)kk * (size_t)M + (size_t)mc;
          hip_bfloat16 t0 = (mc+0<M)? wp[0] : hip_bfloat16(0);
          hip_bfloat16 t1 = (mc+1<M)? wp[1] : hip_bfloat16(0);
          hip_bfloat16 t2 = (mc+2<M)? wp[2] : hip_bfloat16(0);
          hip_bfloat16 t3 = (mc+3<M)? wp[3] : hip_bfloat16(0);
          f0 = bf16_to_f32(t0);
          f1 = bf16_to_f32(t1);
          f2 = bf16_to_f32(t2);
          f3 = bf16_to_f32(t3);
        }
      }
      sB[buf][r_k][c4*4 + 0] = f0;
      sB[buf][r_k][c4*4 + 1] = f1;
      sB[buf][r_k][c4*4 + 2] = f2;
      sB[buf][r_k][c4*4 + 3] = f3;
    }
  };

  // ---- Preload first K-tile
  int buf = 0;
  load_k_tile_into_lds(buf, /*k0=*/0);
  __syncthreads();

  // ---- Main loop over K in tiles of BK
  for (int k0 = 0; k0 < K; k0 += BK) {
    // Optionally prefetch next tile while we compute (ping-pong)
    int next = buf ^ 1;
    if (k0 + BK < K) {
      // Start prefetch by some threads earlier than compute to hide a bit of latency.
      // (On AMD we don't have cp.async; this still helps with ILP/latency overlap.)
      load_k_tile_into_lds(next, k0 + BK);
    }

    __syncthreads(); // ensure sA/sB[buf] are ready before compute

    // ---- Compute this K-tile with MFMA
    // Each wave computes a 16x16 subtile inside the BMxBN block tile.
    // Local coords within the wave's 16x16:
    const int lrow_base = wave_m * 16;
    const int lcol_base = wave_n * 16;

    #pragma unroll
    for (int t = 0; t < (BK / 4); ++t) {
      // Select kk inside the BK-tile via ty (0..3)
      const int kk_in = t * 4 + ty;

      // Fetch A and B operands from LDS
      float a = sA[buf][lrow_base + tx][kk_in];
      float b = sB[buf][kk_in][lcol_base + tx];

      // MFMA 16x16x4: each thread contributes (a,b) for its lane to accumulate 4 outputs
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, acc, 0, 0, 0);
    }

    __syncthreads(); // make sure all finished using sA/sB[buf]

    // Switch buffers
    buf ^= 1;
  }

  // ---- Store accumulators back to Y
  const int out_b_base = b0 + wave_m * 16;
  const int out_m_base = m0 + wave_n * 16;

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    int brow = out_b_base + (ty * 4 + i);
    int mcol = out_m_base + tx;
    if (brow < B && mcol < M) {
      Y[(size_t)brow * (size_t)M + (size_t)mcol] = acc[i];
    }
  }
}

// --------------------------- Host wrapper ---------------------------
static inline void gemm_gpu_batch_bf16(
    float*               __restrict__ Y,  // [B, M]
    const float*         __restrict__ X,  // [B, K]
    const hip_bfloat16*  __restrict__ W,  // [K, M] (BF16, ĐÃ CHUYỂN VỊ)
    int K, int M, int B,
    hipStream_t stream = 0)
{
  if (K <= 0 || M <= 0 || B <= 0) return;

  // Block = 4 waves = (16,4,4)
  dim3 block(16, 4, WAVES_M * WAVES_N);
  dim3 grid(ceil_div(M, BN), ceil_div(B, BM), 1);

  hipLaunchKernelGGL(
      k_gemm_bf16_opt_mfma_2x2waves,
      grid, block,
      /*sharedMemBytes=*/0, stream,
      X, W, Y, K, M, B);
}

__global__ void k_gemm_f32_opt_mfma_2x2waves(
    const float* __restrict__ X,   // [B,K] f32
    const float* __restrict__ W,   // [K,M] f32 (already transposed vs original)
    float*       __restrict__ Y,   // [B,M] f32
    int K, int M, int B)
{
  // Wave-local lane coords for MFMA (16x16x4)
  const int tx = threadIdx.x; // 0..15
  const int ty = threadIdx.y; // 0..3
  const int wz = threadIdx.z; // 0..(WAVES_M*WAVES_N-1)

  // Wave indices inside block tile
  const int wave_m = wz / WAVES_N;  // 0..WAVES_M-1  (rows)
  const int wave_n = wz % WAVES_N;  // 0..WAVES_N-1  (cols)

  // Block origin in Y
  const int m0 = blockIdx.x * BN;   // col start
  const int b0 = blockIdx.y * BM;   // row start

  // Shared memory (double-buffer)
  __shared__ float sA[2][BM][BK + APAD];
  __shared__ float sB[2][BK][BN + BPAD];

  // Accumulator for this thread (4 partial sums)
  v4f32 acc = {0.f, 0.f, 0.f, 0.f};

  // Flattened thread id in block (0..255) to distribute loads
  const int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
  const int tid = wz * (blockDim.y * blockDim.x) + (ty * blockDim.x + tx);

  // ---------- Lambda: cooperative load of one K-tile into LDS ----------
  auto load_k_tile_into_lds = [&](int buf, int k0) {
    // ---- Load X tile: [BM, BK] floats into sA[buf]
    // Vectorized by float4 when aligned; fallback scalar otherwise.
    const int A_F4_W = BK / 4;                // groups of 4 along K
    const int A_TOT_F4 = BM * A_F4_W;         // total float4 to load
    for (int idx = tid; idx < A_TOT_F4; idx += threads_per_block) {
      int r  = idx / A_F4_W;                  // row in [0..BM)
      int c4 = idx % A_F4_W;                  // col-group in [0..BK/4)
      int b_row = b0 + r;
      int kk    = k0 + c4 * 4;

      float4 v = {0,0,0,0};
      if (b_row < B && (kk + 3) < K) {
        const float* gptr = X + (size_t)b_row * (size_t)K + (size_t)kk;
        if ((((uintptr_t)gptr) & 0xF) == 0) {
          v = *reinterpret_cast<const float4*>(gptr);
        } else {
          v.x = gptr[0];
          v.y = gptr[1];
          v.z = gptr[2];
          v.w = gptr[3];
        }
      }
      // Store to LDS with +APAD padding
      sA[buf][r][c4*4 + 0] = v.x;
      sA[buf][r][c4*4 + 1] = v.y;
      sA[buf][r][c4*4 + 2] = v.z;
      sA[buf][r][c4*4 + 3] = v.w;
    }

    // ---- Load W tile: [BK, BN] from W[kk, m] (float) into sB[buf]
    // Vectorized by float4 along M when aligned; fallback scalar otherwise.
    const int B_F4_W = BN / 4;                 // groups of 4 along M
    const int B_TOT_F4 = BK * B_F4_W;          // total float4 to load
    for (int idx = tid; idx < B_TOT_F4; idx += threads_per_block) {
      int r_k = idx / B_F4_W;                  // k in [0..BK)
      int c4  = idx % B_F4_W;                  // col-group in [0..BN/4)
      int kk  = k0 + r_k;                      // global k
      int mc  = m0 + c4 * 4;                   // global m start (4-wide)

      float4 v = {0,0,0,0};
      if (kk < K) {
        const float* gptr = W + (size_t)kk * (size_t)M + (size_t)mc; // row-major [K,M]
        if ((mc + 3) < M) {
          if ((((uintptr_t)gptr) & 0xF) == 0) {
            v = *reinterpret_cast<const float4*>(gptr);
          } else {
            v.x = gptr[0];
            v.y = gptr[1];
            v.z = gptr[2];
            v.w = gptr[3];
          }
        } else {
          // tail-safe (rare for your sizes)
          v.x = (mc+0<M)? gptr[0] : 0.f;
          v.y = (mc+1<M)? gptr[1] : 0.f;
          v.z = (mc+2<M)? gptr[2] : 0.f;
          v.w = (mc+3<M)? gptr[3] : 0.f;
        }
      }
      sB[buf][r_k][c4*4 + 0] = v.x;
      sB[buf][r_k][c4*4 + 1] = v.y;
      sB[buf][r_k][c4*4 + 2] = v.z;
      sB[buf][r_k][c4*4 + 3] = v.w;
    }
  };

  // ---- Preload first K-tile
  int buf = 0;
  load_k_tile_into_lds(buf, /*k0=*/0);
  __syncthreads();

  // ---- Main loop over K in tiles of BK
  for (int k0 = 0; k0 < K; k0 += BK) {
    // Prefetch next tile (ping-pong)
    int next = buf ^ 1;
    if (k0 + BK < K) {
      load_k_tile_into_lds(next, k0 + BK);
    }

    __syncthreads(); // ensure sA/sB[buf] are ready before compute

    // ---- Compute this K-tile with MFMA
    // Each wave computes a 16x16 subtile inside the BMxBN block tile.
    const int lrow_base = wave_m * 16;
    const int lcol_base = wave_n * 16;

    #pragma unroll
    for (int t = 0; t < (BK / 4); ++t) {
      // Select kk inside the BK-tile via ty (0..3)
      const int kk_in = t * 4 + ty;

      // Fetch A and B operands from LDS
      float a = sA[buf][lrow_base + tx][kk_in]; // X tile
      float b = sB[buf][kk_in][lcol_base + tx]; // W tile

      // MFMA 16x16x4: each thread contributes (a,b) and accumulates 4 outputs
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, acc, 0, 0, 0);
    }

    __syncthreads(); // finished using sA/sB[buf]
    buf ^= 1;
  }

  // ---- Store accumulators back to Y
  const int out_b_base = b0 + wave_m * 16;
  const int out_m_base = m0 + wave_n * 16;

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    int brow = out_b_base + (ty * 4 + i);
    int mcol = out_m_base + tx;
    if (brow < B && mcol < M) {
      Y[(size_t)brow * (size_t)M + (size_t)mcol] = acc[i];
    }
  }
}

// --------------------------- Host wrapper ---------------------------
static inline void gemm_gpu_batch_f32W(
    float*       __restrict__ Y,  // [B, M]
    const float* __restrict__ X,  // [B, K]
    const float* __restrict__ W,  // [K, M] (float, ĐÃ CHUYỂN VỊ so với gốc)
    int K, int M, int B,
    hipStream_t stream = 0)
{
  if (K <= 0 || M <= 0 || B <= 0) return;

  // Block = 4 waves = (16,4,4) => 256 threads
  dim3 block(16, 4, WAVES_M * WAVES_N);
  dim3 grid(ceil_div(M, BN), ceil_div(B, BM), 1);

  hipLaunchKernelGGL(
      k_gemm_f32_opt_mfma_2x2waves,
      grid, block,
      /*sharedMemBytes=*/0, stream,
      X, W, Y, K, M, B);
}

// (giữ nguyên các #include, macro BN/BM/BK, APAD/BPAD, WAVES_M/WAVES_N, v4f32,
//  helper bf16_to_f32/bf16x4_u64_to_f4, ceil_div, v.v. như bản của bạn)

// ======================= KERNEL =======================
__global__ void k_gemm_bf16_opt_mfma_2x2waves_2(
    const float* const*  __restrict__ X,  // [B][K] f32  // [CHANGED] was: const float* X
    const hip_bfloat16*  __restrict__ W,  // [K,M] bf16 (transposed already)
    float*               __restrict__ Y,  // [B,M] f32
    int K, int M, int B)
{
  // Wave-local lane coords for MFMA (16x16x4)
  const int tx = threadIdx.x; // 0..15
  const int ty = threadIdx.y; // 0..3
  const int wz = threadIdx.z; // 0..(WAVES_M*WAVES_N-1)

  // Wave indices inside block tile
  const int wave_m = wz / WAVES_N;  // 0..WAVES_M-1  (rows)
  const int wave_n = wz % WAVES_N;  // 0..WAVES_N-1  (cols)

  // Block origin in Y
  const int m0 = blockIdx.x * BN;   // col start
  const int b0 = blockIdx.y * BM;   // row start

  // Shared memory (double-buffer)
  __shared__ float sA[2][BM][BK + APAD];
  __shared__ float sB[2][BK][BN + BPAD];

  // Accumulator for this thread (4 partial sums)
  v4f32 acc = {0.f, 0.f, 0.f, 0.f};

  // Flattened thread id in block (0..255) to distribute loads
  const int tid = (wz * (blockDim.y * blockDim.x)) + (ty * blockDim.x + tx);

  // ---------- Lambda: cooperative load of one K-tile into LDS ----------
  auto load_k_tile_into_lds = [&](int buf, int k0) {
    // ---- Load X tile: [BM, BK] floats into sA[buf]
    // Vectorized by float4 when aligned; fallback scalar otherwise.
    const int A_F4_W   = BK / 4;           // groups of 4 along K
    const int A_TOT_F4 = BM * A_F4_W;      // total float4 to load
    for (int idx = tid; idx < A_TOT_F4; idx += blockDim.x * blockDim.y * blockDim.z) {
      int r  = idx / A_F4_W;               // row in [0..BM)
      int c4 = idx % A_F4_W;               // col-group in [0..BK/4)
      int b_row = b0 + r;
      int kk    = k0 + c4 * 4;

      float4 v = {0,0,0,0};
      if (b_row < B && (kk + 3) < K) {
        const float* base = X[b_row];                      // [CHANGED] was: X + b_row*K
        const float* gptr = base ? (base + kk) : nullptr;  // [CHANGED]
        if (gptr) {
          // Vectorized if 16B aligned, else scalar gather
          if ((((uintptr_t)gptr) & 0xF) == 0) {
            v = *reinterpret_cast<const float4*>(gptr);
          } else {
            v.x = gptr[0];
            v.y = gptr[1];
            v.z = gptr[2];
            v.w = gptr[3];
          }
        }
      }
      // Store to LDS with +APAD padding
      sA[buf][r][c4*4 + 0] = v.x;
      sA[buf][r][c4*4 + 1] = v.y;
      sA[buf][r][c4*4 + 2] = v.z;
      sA[buf][r][c4*4 + 3] = v.w;
    }

    // ---- Load W tile: [BK, BN] from W[kk, m] (BF16) -> float into sB[buf]
    // Vectorized by 4×bf16 (uint64)
    const int B_BF4_W   = BN / 4;          // groups of 4 along M
    const int B_TOT_BF4 = BK * B_BF4_W;    // total bf16x4 packs to load
    for (int idx = tid; idx < B_TOT_BF4; idx += blockDim.x * blockDim.y * blockDim.z) {
      int r_k = idx / B_BF4_W;             // k in [0..BK)
      int c4  = idx % B_BF4_W;             // col-group in [0..BN/4)
      int kk  = k0 + r_k;                  // global k
      int mc  = m0 + c4 * 4;               // global m start (4-wide)

      float f0=0,f1=0,f2=0,f3=0;
      if (kk < K) {
        if ((mc + 3) < M) {
          const uint64_t* gptr = reinterpret_cast<const uint64_t*>(
              W + (size_t)kk * (size_t)M + (size_t)mc);
          uint64_t u = *gptr; // contiguous across M (row-major [K,M])
          bf16x4_u64_to_f4(u, f0, f1, f2, f3);
        } else {
          const hip_bfloat16* wp = W + (size_t)kk * (size_t)M + (size_t)mc;
          hip_bfloat16 t0 = (mc+0<M)? wp[0] : hip_bfloat16(0);
          hip_bfloat16 t1 = (mc+1<M)? wp[1] : hip_bfloat16(0);
          hip_bfloat16 t2 = (mc+2<M)? wp[2] : hip_bfloat16(0);
          hip_bfloat16 t3 = (mc+3<M)? wp[3] : hip_bfloat16(0);
          f0 = bf16_to_f32(t0);
          f1 = bf16_to_f32(t1);
          f2 = bf16_to_f32(t2);
          f3 = bf16_to_f32(t3);
        }
      }
      sB[buf][r_k][c4*4 + 0] = f0;
      sB[buf][r_k][c4*4 + 1] = f1;
      sB[buf][r_k][c4*4 + 2] = f2;
      sB[buf][r_k][c4*4 + 3] = f3;
    }
  };

  // ---- Preload first K-tile
  int buf = 0;
  load_k_tile_into_lds(buf, /*k0=*/0);
  __syncthreads();

  // ---- Main loop over K in tiles of BK
  for (int k0 = 0; k0 < K; k0 += BK) {
    int next = buf ^ 1;
    if (k0 + BK < K) {
      load_k_tile_into_lds(next, k0 + BK);
    }

    __syncthreads(); // ensure sA/sB[buf] are ready before compute

    // ---- Compute this K-tile with MFMA (each wave does 16x16)
    const int lrow_base = wave_m * 16;
    const int lcol_base = wave_n * 16;

    #pragma unroll
    for (int t = 0; t < (BK / 4); ++t) {
      const int kk_in = t * 4 + ty;

      float a = sA[buf][lrow_base + tx][kk_in];
      float b = sB[buf][kk_in][lcol_base + tx];

      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, acc, 0, 0, 0);
    }

    __syncthreads(); // finished using sA/sB[buf]
    buf ^= 1;
  }

  // ---- Store accumulators back to Y
  const int out_b_base = b0 + wave_m * 16;
  const int out_m_base = m0 + wave_n * 16;

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    int brow = out_b_base + (ty * 4 + i);
    int mcol = out_m_base + tx;
    if (brow < B && mcol < M) {
      Y[(size_t)brow * (size_t)M + (size_t)mcol] = acc[i];
    }
  }
}

// --------------------------- Host wrapper ---------------------------
// Lưu ý: X_dev là con trỏ thiết bị tới mảng B con trỏ thiết bị mỗi hàng K phần tử.
static inline void gemm_gpu_batch_bf16_2(
    float*               __restrict__ Y,       // [B, M]
    const float* const*  __restrict__ X_dev,   // [B][K]        // [CHANGED] was: const float* X
    const hip_bfloat16*  __restrict__ W,       // [K, M] (BF16, ĐÃ CHUYỂN VỊ)
    int K, int M, int B,
    hipStream_t stream = 0)
{
  if (K <= 0 || M <= 0 || B <= 0) return;

  // Block = 4 waves = (16,4,4)
  dim3 block(16, 4, WAVES_M * WAVES_N);
  dim3 grid(ceil_div(M, BN), ceil_div(B, BM), 1);

  hipLaunchKernelGGL(
      k_gemm_bf16_opt_mfma_2x2waves_2,
      grid, block,
      /*sharedMemBytes=*/0, stream,
      X_dev, W, Y, K, M, B); // [CHANGED] X_dev thay cho X phẳng
}
