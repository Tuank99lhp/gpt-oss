// template<int BM, int BN, int BK, int WAVES_M, int WAVES_N, int APAD, int BPAD>
// __global__ void k_gemm_bf16core_yfp32_2x2waves(const hip_bfloat16* __restrict__ X,
//                                                const hip_bfloat16* __restrict__ W,
//                                                float*              __restrict__ Y,
//                                                int K, int M, int B) {
//   const int tx = threadIdx.x;
//   const int ty = threadIdx.y;
//   const int wz = threadIdx.z;

//   const int wave_m = wz / WAVES_N;
//   const int wave_n = wz % WAVES_N;

//   const int m0 = blockIdx.x * BN;
//   const int b0 = blockIdx.y * BM;

//   __shared__ hip_bfloat16 sA[2][BM][BK + APAD];
//   __shared__ hip_bfloat16 sB[2][BK][BN + BPAD];

//   v4f32 acc = {0.f, 0.f, 0.f, 0.f};

//   const int nthreads = blockDim.x * blockDim.y * blockDim.z;
//   const int tid      = wz * (blockDim.y * blockDim.x) + (ty * blockDim.x + tx);

//   auto load_tile = [&](int buf, int k0) {
//     const int A_F4  = BK / 4;
//     const int A_TOT = BM * A_F4;

//     for (int idx = tid; idx < A_TOT; idx += nthreads) {
//       const int r   = idx / A_F4;
//       const int c4  = idx % A_F4;
//       const int br  = b0 + r;
//       const int kk  = k0 + c4 * 4;

//       hip_bfloat16 a0{}, a1{}, a2{}, a3{};

//       if (br < B && (kk + 3) < K) {
//         const hip_bfloat16* g = X + br * K + kk;
//         gld_bf16x4(g, a0, a1, a2, a3);
//       }

//       sA[buf][r][c4 * 4 + 0] = a0;
//       sA[buf][r][c4 * 4 + 1] = a1;
//       sA[buf][r][c4 * 4 + 2] = a2;
//       sA[buf][r][c4 * 4 + 3] = a3;
//     }

//     const int B_F4  = BN / 4;
//     const int B_TOT = BK * B_F4;

//     for (int idx = tid; idx < B_TOT; idx += nthreads) {
//       const int rk  = idx / B_F4;
//       const int c4  = idx % B_F4;
//       const int kk  = k0 + rk;
//       const int mc  = m0 + c4 * 4;

//       hip_bfloat16 b0v{}, b1v{}, b2v{}, b3v{};

//       if (kk < K) {
//         const hip_bfloat16* g = W + kk * M + mc;

//         if ((mc + 3) < M) {
//           gld_bf16x4(g, b0v, b1v, b2v, b3v);
//         } else {
//           if (mc + 0 < M) b0v = g[0];
//           if (mc + 1 < M) b1v = g[1];
//           if (mc + 2 < M) b2v = g[2];
//           if (mc + 3 < M) b3v = g[3];
//         }
//       }

//       sB[buf][rk][c4 * 4 + 0] = b0v;
//       sB[buf][rk][c4 * 4 + 1] = b1v;
//       sB[buf][rk][c4 * 4 + 2] = b2v;
//       sB[buf][rk][c4 * 4 + 3] = b3v;
//     }
//   };

//   int buf = 0;
//   load_tile(buf, 0);
//   __syncthreads();

//   const int lrow0 = wave_m * 16;
//   const int lcol0 = wave_n * 16;

//   for (int k0 = 0; k0 < K; k0 += BK) {
//     const int nxt = buf ^ 1;

//     if (k0 + BK < K) {
//       load_tile(nxt, k0 + BK);
//     }

//     __syncthreads();

//     #pragma unroll
//     for (int t = 0; t < (BK / 16); ++t) {
//       const int kbase = t * 16 + ty * 4;
//       const int base  = kbase; 

//       // NẠP 4 phần tử liên tiếp cho A và B từ shared memory
//       const hip_bfloat16 a0 = sA[buf][lrow0 + tx][base + 0];
//       const hip_bfloat16 a1 = sA[buf][lrow0 + tx][base + 1];
//       const hip_bfloat16 a2 = sA[buf][lrow0 + tx][base + 2];
//       const hip_bfloat16 a3 = sA[buf][lrow0 + tx][base + 3];

//       const hip_bfloat16 b0v = sB[buf][base + 0][lcol0 + tx];
//       const hip_bfloat16 b1v = sB[buf][base + 1][lcol0 + tx];
//       const hip_bfloat16 b2v = sB[buf][base + 2][lcol0 + tx];
//       const hip_bfloat16 b3v = sB[buf][base + 3][lcol0 + tx];

//       // ĐÓNG GÓI thành short4 (v4i16) đúng kiểu builtin yêu cầu
//       const v4i16 av = pack_bf16x4_vec(a0, a1, a2, a3);
//       const v4i16 bv = pack_bf16x4_vec(b0v, b1v, b2v, b3v);

//       // MFMA bf16 -> f32: acc là v4f32
//       acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, acc, 0, 0, 0);
//     }

//     __syncthreads();
//     buf ^= 1;
//   }

//   const int out_b0 = b0 + wave_m * 16;
//   const int out_m0 = m0 + wave_n * 16;

//   #pragma unroll
//   for (int i = 0; i < 4; ++i) {
//     const int brow = out_b0 + (ty * 4 + i);
//     const int mcol = out_m0 + tx;

//     if (brow < B && mcol < M) {
//       Y[brow * M + mcol] = acc[i];
//     }
//   }
// }

// template<int BM = 64,
//          int BN = 64,
//          int BK = 112,
//          int WAVES_M = 4,
//          int WAVES_N = 4,
//          int APAD = 1,
//          int BPAD = 1>
// static inline void gemm_gpu_batch_bf16core_yfp32(float*              __restrict__ Y,
//                                                  const hip_bfloat16* __restrict__ X,
//                                                  const hip_bfloat16* __restrict__ W,
//                                                  int K, int M, int B,
//                                                  hipStream_t stream = 0) {
//   if (K <= 0 || M <= 0 || B <= 0) {
//     return;
//   }

//   static_assert(BM == WAVES_M * 16, "BM must equal WAVES_M*16");
//   static_assert(BN == WAVES_N * 16, "BN must equal WAVES_N*16");
//   static_assert(BK % 16 == 0,       "BK must be a multiple of 16");

//   const dim3 block(16, 4, WAVES_M * WAVES_N);
//   const dim3 grid(ceil_div(M, BN), ceil_div(B, BM), 1);

//   hipLaunchKernelGGL(
//     HIP_KERNEL_NAME(k_gemm_bf16core_yfp32_2x2waves<BM, BN, BK, WAVES_M, WAVES_N, APAD, BPAD>),
//     grid, block, 0, stream,
//     X, W, Y, K, M, B
//   );
// }

template<int BM, int BN, int BK, int WAVES_M, int WAVES_N>
__global__ void k_gemm_bf16core_yfp32_2x2waves(const hip_bfloat16* __restrict__ X,
                                               const hip_bfloat16* __restrict__ W,
                                               float*              __restrict__ Y,
                                               int K, int M, int B,
                                               const float*        __restrict__ bias)
{
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int wz = threadIdx.z;

  const int wave_m = wz / WAVES_N;
  const int wave_n = wz % WAVES_N;

  const int m0 = blockIdx.x * BN;
  const int b0 = blockIdx.y * BM;

  __shared__ __align__(64) hip_bfloat16 sB[2][BK][BN];

  const int lrow0 = wave_m * 16;
  const int lcol0 = wave_n * 16;

  constexpr int SUB_TILES_N = BN / (16 * WAVES_N);

  v4f32 acc[SUB_TILES_N];
  #pragma unroll
  for (int j = 0; j < SUB_TILES_N; ++j) acc[j] = (v4f32){0.f,0.f,0.f,0.f};

  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  const int tid      = wz * (blockDim.x * blockDim.y) + (ty * blockDim.x + tx);

  auto load_B_tile = [&](int buf, int k0) {
    const int B_F4  = BN / 4;
    const int B_TOT = BK * B_F4;
    for (int idx = tid; idx < B_TOT; idx += nthreads) {
      const int rk  = idx / B_F4;
      const int c4  = idx % B_F4;
      const int kk  = k0 + rk;
      const int mc  = m0 + c4 * 4;

      hip_bfloat16 b0v{}, b1v{}, b2v{}, b3v{};
      if (kk < K) {
        const hip_bfloat16* g = W + kk * M + mc;
        if ((mc + 3) < M) {
          gld_bf16x4(g, b0v, b1v, b2v, b3v);
        } else {
          if (mc + 0 < M) b0v = g[0];
          if (mc + 1 < M) b1v = g[1];
          if (mc + 2 < M) b2v = g[2];
          if (mc + 3 < M) b3v = g[3];
        }
      }

      sB[buf][rk][c4 * 4 + 0] = b0v;
      sB[buf][rk][c4 * 4 + 1] = b1v;
      sB[buf][rk][c4 * 4 + 2] = b2v;
      sB[buf][rk][c4 * 4 + 3] = b3v;
    }
  };

  int buf = 0;
  load_B_tile(buf, 0);
  __syncthreads();

  const int brow_base = b0 + lrow0 + tx;

  for (int k0 = 0; k0 < K; k0 += BK) {
    const int nxt = buf ^ 1;
    if (k0 + BK < K) {
      load_B_tile(nxt, k0 + BK);
    }

    __syncthreads();

    #pragma unroll
    for (int t = 0; t < (BK / 16); ++t) {
      const int base_rel = t * 16 + ty * 4;
      const int base_abs = k0 + base_rel;

      hip_bfloat16 a0{}, a1{}, a2{}, a3{};
      if (brow_base < B && base_abs < K) {
        const int valid = min(4, K - base_abs);
        const hip_bfloat16* gA = X + brow_base * K + base_abs;
        if (valid == 4) {
          gld_bf16x4(gA, a0, a1, a2, a3);
        } else {
          if (valid >= 1) a0 = gA[0];
          if (valid >= 2) a1 = gA[1];
          if (valid >= 3) a2 = gA[2];
        }
      }
      const v4i16 av = pack_bf16x4_vec(a0, a1, a2, a3);

      #pragma unroll
      for (int j = 0; j < SUB_TILES_N; ++j) {
        const int ncol = lcol0 + j * 16 + tx;

        const hip_bfloat16 b0v = sB[buf][base_rel + 0][ncol];
        const hip_bfloat16 b1v = sB[buf][base_rel + 1][ncol];
        const hip_bfloat16 b2v = sB[buf][base_rel + 2][ncol];
        const hip_bfloat16 b3v = sB[buf][base_rel + 3][ncol];
        const v4i16        bv  = pack_bf16x4_vec(b0v, b1v, b2v, b3v);

        acc[j] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, acc[j], 0, 0, 0);
      }
    }

    __syncthreads();
    buf ^= 1;
  }

  const int out_b0 = b0 + wave_m * 16;
  const int out_m0 = m0 + lcol0;

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int brow = out_b0 + (ty * 4 + i);
    if (brow >= B) continue;
    
    #pragma unroll
    for (int j = 0; j < SUB_TILES_N; ++j) {
      const int mcol = out_m0 + j * 16 + tx;
      if (mcol < M) {
        Y[brow * M + mcol] = acc[j][i] + (bias ? bias[mcol] : 0.f);
      }
    }
  }
}

template<int BM, int BN, int BK, int WAVES_M, int WAVES_N>
static inline void gemm_gpu_batch_bf16core_yfp32(float*              __restrict__ Y,
                                                 const hip_bfloat16* __restrict__ X,
                                                 const hip_bfloat16* __restrict__ W,
                                                 int K, int M, int B,
                                                 const float*        __restrict__ bias = nullptr,
                                                 hipStream_t stream = 0)
{
  if (K <= 0 || M <= 0 || B <= 0) return;

  static_assert(BM == WAVES_M * 16, "BM must equal WAVES_M*16");
  static_assert(BN % (16 * WAVES_N) == 0, "BN must be multiple of 16*WAVES_N");
  static_assert(BK % 16 == 0, "BK must be multiple of 16");

  constexpr int SUB_TILES_N = BN / (16 * WAVES_N);
  static_assert(SUB_TILES_N >= 1 && SUB_TILES_N <= 4, "BN/(16*WAVES_N) must be in [1,4]");

  const dim3 block(16, 4, WAVES_M * WAVES_N);
  const dim3 grid(ceil_div(M, BN), ceil_div(B, BM), 1);

  hipLaunchKernelGGL(
    HIP_KERNEL_NAME((k_gemm_bf16core_yfp32_2x2waves<BM, BN, BK, WAVES_M, WAVES_N>)),
    grid, block, 0, stream,
    X, W, Y, K, M, B, bias
  );
}
