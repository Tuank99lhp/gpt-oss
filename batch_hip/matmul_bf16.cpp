template<int BM, int BN, int BK, int WAVES_M, int WAVES_N, int APAD, int BPAD>
__global__ void k_gemm_bf16core_ybf16_2x2waves(const hip_bfloat16* __restrict__ X,
                                               const hip_bfloat16* __restrict__ W,
                                               hip_bfloat16*       __restrict__ Y,
                                               int K,
                                               int M,
                                               int B) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int wz = threadIdx.z;

  const int wave_m = wz / WAVES_N;
  const int wave_n = wz % WAVES_N;

  const int m0 = blockIdx.x * BN;
  const int b0 = blockIdx.y * BM;

  __shared__ hip_bfloat16 sA[2][BM][BK + APAD];
  __shared__ hip_bfloat16 sB[2][BK][BN + BPAD];

  v4f32 acc = {0.f, 0.f, 0.f, 0.f};

  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  const int tid      = wz * (blockDim.y * blockDim.x) + (ty * blockDim.x + tx);

  auto load_tile = [&](int buf, int k0) {
    const int A_F4  = BK / 4;
    const int A_TOT = BM * A_F4;

    for (int idx = tid; idx < A_TOT; idx += nthreads) {
      const int r   = idx / A_F4;
      const int c4  = idx % A_F4;
      const int br  = b0 + r;
      const int kk  = k0 + c4 * 4;

      hip_bfloat16 a0{};
      hip_bfloat16 a1{};
      hip_bfloat16 a2{};
      hip_bfloat16 a3{};

      if (br < B && (kk + 3) < K) {
        const hip_bfloat16* g = X + br * K + kk;
        gld_bf16x4(g, a0, a1, a2, a3);
      }

      sA[buf][r][c4 * 4 + 0] = a0;
      sA[buf][r][c4 * 4 + 1] = a1;
      sA[buf][r][c4 * 4 + 2] = a2;
      sA[buf][r][c4 * 4 + 3] = a3;
    }

    const int B_F4  = BN / 4;
    const int B_TOT = BK * B_F4;

    for (int idx = tid; idx < B_TOT; idx += nthreads) {
      const int rk  = idx / B_F4;
      const int c4  = idx % B_F4;
      const int kk  = k0 + rk;
      const int mc  = m0 + c4 * 4;

      hip_bfloat16 b0v{};
      hip_bfloat16 b1v{};
      hip_bfloat16 b2v{};
      hip_bfloat16 b3v{};

      if (kk < K) {
        const hip_bfloat16* g = W + kk * M + mc;

        if ((mc + 3) < M) {
          gld_bf16x4(g, b0v, b1v, b2v, b3v);
        } else {
          if (mc + 0 < M) {
            b0v = g[0];
          }
          if (mc + 1 < M) {
            b1v = g[1];
          }
          if (mc + 2 < M) {
            b2v = g[2];
          }
          if (mc + 3 < M) {
            b3v = g[3];
          }
        }
      }

      sB[buf][rk][c4 * 4 + 0] = b0v;
      sB[buf][rk][c4 * 4 + 1] = b1v;
      sB[buf][rk][c4 * 4 + 2] = b2v;
      sB[buf][rk][c4 * 4 + 3] = b3v;
    }
  };

  int buf = 0;
  load_tile(buf, 0);
  __syncthreads();

  const int lrow0 = wave_m * 16;
  const int lcol0 = wave_n * 16;

  for (int k0 = 0; k0 < K; k0 += BK) {
    const int nxt = buf ^ 1;

    if (k0 + BK < K) {
      load_tile(nxt, k0 + BK);
    }

    __syncthreads();

    #pragma unroll
    for (int t = 0; t < (BK / 16); ++t) {
      const int kbase = t * 16 + ty * 4;      // mỗi ty lấy một nhóm 4 theo K
      const int base  = kbase; 

      // NẠP 4 phần tử liên tiếp cho A và B từ shared memory
      const hip_bfloat16 a0 = sA[buf][lrow0 + tx][base + 0];
      const hip_bfloat16 a1 = sA[buf][lrow0 + tx][base + 1];
      const hip_bfloat16 a2 = sA[buf][lrow0 + tx][base + 2];
      const hip_bfloat16 a3 = sA[buf][lrow0 + tx][base + 3];

      const hip_bfloat16 b0v = sB[buf][base + 0][lcol0 + tx];
      const hip_bfloat16 b1v = sB[buf][base + 1][lcol0 + tx];
      const hip_bfloat16 b2v = sB[buf][base + 2][lcol0 + tx];
      const hip_bfloat16 b3v = sB[buf][base + 3][lcol0 + tx];

      // ĐÓNG GÓI thành short4 (v4i16) đúng kiểu builtin yêu cầu
      const v4i16 av = pack_bf16x4_vec(a0, a1, a2, a3);
      const v4i16 bv = pack_bf16x4_vec(b0v, b1v, b2v, b3v);

      // MFMA bf16 -> f32: acc là v4f32
      acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, acc, 0, 0, 0);
    }

    __syncthreads();
    buf ^= 1;
  }

  const int out_b0 = b0 + wave_m * 16;
  const int out_m0 = m0 + wave_n * 16;

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int brow = out_b0 + (ty * 4 + i);
    const int mcol = out_m0 + tx;

    if (brow < B && mcol < M) {
      const float v = acc[i];
      Y[brow * M + mcol] = f32_to_bf16(v);
    }
  }
}

template<int BM = 64,
         int BN = 64,
         int BK = 64,
         int WAVES_M = 4,
         int WAVES_N = 4,
         int APAD = 1,
         int BPAD = 1>
static inline void gemm_gpu_batch_bf16core_ybf16(hip_bfloat16*       __restrict__ Y,
                                                 const hip_bfloat16* __restrict__ X,
                                                 const hip_bfloat16* __restrict__ W,
                                                 int K, int M, int B,
                                                 hipStream_t stream = 0) {
  if (K <= 0 || M <= 0 || B <= 0) {
    return;
  }

  static_assert(BM == WAVES_M * 16, "BM must equal WAVES_M*16");
  static_assert(BN == WAVES_N * 16, "BN must equal WAVES_N*16");
  static_assert(BK % 16 == 0,       "BK must be a multiple of 16");

  const dim3 block(16, 4, WAVES_M * WAVES_N);
  const dim3 grid(ceil_div(M, BN), ceil_div(B, BM), 1);

  hipLaunchKernelGGL(
    HIP_KERNEL_NAME(k_gemm_bf16core_ybf16_2x2waves<BM, BN, BK, WAVES_M, WAVES_N, APAD, BPAD>),
    grid, block, 0, stream,
    X, W, Y, K, M, B
  );
}
