// gemv_batched_wave_row.cu
// Batched GEMV tối ưu dựa trên GEMV đơn-mẫu: wave-per-row + LDS ping-pong.
// Y[b, m] = sum_k W[m, k] * X[b, k]

// ================= Wave-per-row batched (no split-K) =================
// blockDim = (waves_per_block * WARP_SIZE, TB)
// grid     = (ceil(M / waves_per_block), ceil(B / TB))
// shared   = 2 * TB * TK * sizeof(float)  (ping-pong X tiles)
template<int TK=1024, int TB=4, bool VEC4=true>
__global__ void k_gemv_batched_wave_row_shared(const float* __restrict__ W, // [M,K]
                                               const float* __restrict__ X, // [B,K]
                                               float*       __restrict__ Y, // [B,M]
                                               int K, int M, int B)
{
  extern __shared__ float s[];
  float* X0 = s;
  float* X1 = s + (size_t)TB * TK;

  const int wid  = warp_id();        // 0..waves_per_block-1
  const int lane = lane_id();        // 0..WARP_SIZE-1
  const int waves_per_block = blockDim.x / WARP_SIZE;

  const int b0 = blockIdx.y * TB;
  const int m0 = blockIdx.x * waves_per_block;

  const int ty = threadIdx.y;        // 0..TB-1 (sample lane)
  const int b  = b0 + ty;            // global sample
  const int m  = m0 + wid;           // global row (per-wave)

  const bool active = (b < B) && (m < M);
  const bool smem_aligned16 = (((uintptr_t)s & 0xF) == 0);
  const bool stride16 = ((K & 3) == 0);
  const bool W_base16 = (((uintptr_t)W & 0xF) == 0);
  const bool X_base16 = (((uintptr_t)X & 0xF) == 0);

  // Cooperative loading helpers
  const int threadsPerBlock = blockDim.x * blockDim.y;
  const int tidLinear = threadIdx.y * blockDim.x + threadIdx.x;

  float acc = 0.0f;
  int   buf = 0;

  // ---- Preload first tile of X into X0 ----
  int k0 = 0;
  {
    const int k_lim = dmin(TK, K - k0);
    const bool vec4_ok = VEC4 && stride16 && ((k0 & 3) == 0) && (k_lim >= 4) && X_base16;

    if (vec4_ok){
      const int cols4  = (k_lim >> 2);
      const int total4 = TB * cols4;
      for (int idx4 = tidLinear; idx4 < total4; idx4 += threadsPerBlock){
        int tb  = idx4 / cols4;
        int tk4 = idx4 - tb * cols4;
        int gb  = b0 + tb;
        int gk  = k0 + (tk4 << 2);
        Float4 v = {0,0,0,0};
        if (gb < B){
          const Float4* px = reinterpret_cast<const Float4*>(
              X + (size_t)gb * K + gk);
          v = *px;
        }
        // store SCALAR to LDS to avoid alignment issues row-by-row
        const int base = tb * TK + (tk4<<2);
        if (0 <  k_lim) X0[base + 0] = v.x;
        if (1 <  k_lim) X0[base + 1] = v.y;
        if (2 <  k_lim) X0[base + 2] = v.z;
        if (3 <  k_lim) X0[base + 3] = v.w;
      }
      // tail (should be rare)
      const int vec_end = (cols4 << 2);
      const int tail    = k_lim - vec_end;
      if (tail){
        const int span = TB * tail;
        for (int idx = tidLinear; idx < span; idx += threadsPerBlock){
          int tb = idx / tail;
          int tk = idx - tb * tail;
          int gb = b0 + tb;
          int gk = k0 + vec_end + tk;
          X0[tb * TK + (vec_end + tk)] = (gb < B) ? X[(size_t)gb * K + gk] : 0.0f;
        }
      }
    } else {
      const int span = TB * k_lim;
      for (int idx = tidLinear; idx < span; idx += threadsPerBlock){
        int tb = idx / k_lim;
        int tk = idx - tb * k_lim;
        int gb = b0 + tb;
        int gk = k0 + tk;
        X0[tb * TK + tk] = (gb < B) ? X[(size_t)gb * K + gk] : 0.0f;
      }
    }
  }
  __syncthreads();

  // ---- Main K loop (ping-pong X) ----
  for (; k0 < K; k0 += TK, buf ^= 1){
    const int k_lim = dmin(TK, K - k0);
    float* x_cur = (buf == 0 ? X0 : X1);
    float* x_nxt = (buf == 0 ? X1 : X0);

    if (active){
      const float* __restrict__ wrow = W + (size_t)m * K + k0;
      const float* __restrict__ xrow = x_cur + (size_t)ty * TK;

      if (VEC4 && smem_aligned16 && W_base16 && stride16 && ((k0 & 3) == 0) && (k_lim >= 4)){
        const int vec_end = (k_lim >> 2) << 2;
        for (int jj = lane * 4; jj < vec_end; jj += 4 * WARP_SIZE){
          const Float4* __restrict__ w4 = reinterpret_cast<const Float4*>(wrow + jj);
          const Float4* __restrict__ x4 = reinterpret_cast<const Float4*>(xrow + jj);
          Float4 a = *w4;
          Float4 b = *x4; // LDS vector read is OK if base is 16B and TK%4==0
          acc = fmaf(a.x, b.x, acc);
          acc = fmaf(a.y, b.y, acc);
          acc = fmaf(a.z, b.z, acc);
          acc = fmaf(a.w, b.w, acc);
        }
        for (int jj = ((k_lim>>2)<<2) + lane; jj < k_lim; jj += WARP_SIZE){
          acc = fmaf(wrow[jj], xrow[jj], acc);
        }
      } else {
        #pragma unroll 4
        for (int jj = lane; jj < k_lim; jj += WARP_SIZE){
          acc = fmaf(wrow[jj], xrow[jj], acc);
        }
      }
    }

    // ---- Preload next X tile (cooperative) ----
    const int next_k0 = k0 + TK;
    if (next_k0 < K){
      const int next_lim = dmin(TK, K - next_k0);
      const bool vec4_ok = VEC4 && stride16 && ((next_k0 & 3) == 0) && (next_lim >= 4) && X_base16;

      if (vec4_ok){
        const int cols4  = (next_lim >> 2);
        const int total4 = TB * cols4;
        for (int idx4 = tidLinear; idx4 < total4; idx4 += threadsPerBlock){
          int tb  = idx4 / cols4;
          int tk4 = idx4 - tb * cols4;
          int gb  = b0 + tb;
          int gk  = next_k0 + (tk4 << 2);
          Float4 v = {0,0,0,0};
          if (gb < B){
            const Float4* px = reinterpret_cast<const Float4*>(
                X + (size_t)gb * K + gk);
            v = *px;
          }
          const int base = tb * TK + (tk4<<2);
          if (0 < next_lim) x_nxt[base + 0] = v.x;
          if (1 < next_lim) x_nxt[base + 1] = v.y;
          if (2 < next_lim) x_nxt[base + 2] = v.z;
          if (3 < next_lim) x_nxt[base + 3] = v.w;
        }
        const int vec_end = (cols4 << 2);
        const int tail    = next_lim - vec_end;
        if (tail){
          const int span = TB * tail;
          for (int idx = tidLinear; idx < span; idx += threadsPerBlock){
            int tb = idx / tail;
            int tk = idx - tb * tail;
            int gb = b0 + tb;
            int gk = next_k0 + vec_end + tk;
            x_nxt[tb * TK + (vec_end + tk)] = (gb < B) ? X[(size_t)gb * K + gk] : 0.0f;
          }
        }
      } else {
        const int span = TB * next_lim;
        for (int idx = tidLinear; idx < span; idx += threadsPerBlock){
          int tb = idx / next_lim;
          int tk = idx - tb * next_lim;
          int gb = b0 + tb;
          int gk = next_k0 + tk;
          x_nxt[tb * TK + tk] = (gb < B) ? X[(size_t)gb * K + gk] : 0.0f;
        }
      }
    }

    __syncthreads();
  }

  // ---- Write result ----
  float sum = warp_reduce_sum(acc);
  if (lane == 0 && active){
    Y[(size_t)b * M + m] = sum;
  }
}

// ================= Split-K batched (atomicAdd to Y) =================
// grid.z = splitK; mỗi block xử lý một partition k_begin..k_end
template<int TK=1024, int TB=4, bool VEC4=true>
__global__ void k_gemv_batched_wave_row_shared_splitK(const float* __restrict__ W,
                                                      const float* __restrict__ X,
                                                      float*       __restrict__ Y, // atomicAdd
                                                      int K, int M, int B, int splitK)
{
  extern __shared__ float s[];
  float* X0 = s;
  float* X1 = s + (size_t)TB * TK;

  const int wid  = warp_id();
  const int lane = lane_id();
  const int waves_per_block = blockDim.x / WARP_SIZE;

  const int b0 = blockIdx.y * TB;
  const int m0 = blockIdx.x * waves_per_block;
  const int part = blockIdx.z;

  const int span    = dceil_div(K, splitK);
  const int k_begin = part * span;
  const int k_end   = dmin(K, k_begin + span);
  if (k_begin >= k_end) return;

  const int ty = threadIdx.y;
  const int b  = b0 + ty;
  const int m  = m0 + wid;

  const bool active = (b < B) && (m < M);
  const bool smem_aligned16 = (((uintptr_t)s & 0xF) == 0);
  const bool stride16 = ((K & 3) == 0);
  const bool W_base16 = (((uintptr_t)W & 0xF) == 0);
  const bool X_base16 = (((uintptr_t)X & 0xF) == 0);

  const int threadsPerBlock = blockDim.x * blockDim.y;
  const int tidLinear = threadIdx.y * blockDim.x + threadIdx.x;

  float acc = 0.0f;
  int   buf = 0;

  // Preload first X tile
  int k0 = k_begin;
  {
    const int k_lim = dmin(TK, k_end - k0);
    const bool vec4_ok = VEC4 && stride16 && ((k0 & 3) == 0) && (k_lim >= 4) && X_base16;

    if (vec4_ok){
      const int cols4  = (k_lim >> 2);
      const int total4 = TB * cols4;
      for (int idx4 = tidLinear; idx4 < total4; idx4 += threadsPerBlock){
        int tb  = idx4 / cols4;
        int tk4 = idx4 - tb * cols4;
        int gb  = b0 + tb;
        int gk  = k0 + (tk4 << 2);
        Float4 v = {0,0,0,0};
        if (gb < B){
          const Float4* px = reinterpret_cast<const Float4*>(
              X + (size_t)gb * K + gk);
          v = *px;
        }
        const int base = tb * TK + (tk4<<2);
        if (0 <  k_lim) X0[base + 0] = v.x;
        if (1 <  k_lim) X0[base + 1] = v.y;
        if (2 <  k_lim) X0[base + 2] = v.z;
        if (3 <  k_lim) X0[base + 3] = v.w;
      }
      const int vec_end = (cols4 << 2);
      const int tail    = k_lim - vec_end;
      if (tail){
        const int span2 = TB * tail;
        for (int idx = tidLinear; idx < span2; idx += threadsPerBlock){
          int tb = idx / tail;
          int tk = idx - tb * tail;
          int gb = b0 + tb;
          int gk = k0 + vec_end + tk;
          X0[tb * TK + (vec_end + tk)] = (gb < B) ? X[(size_t)gb * K + gk] : 0.0f;
        }
      }
    } else {
      const int span2 = TB * k_lim;
      for (int idx = tidLinear; idx < span2; idx += threadsPerBlock){
        int tb = idx / k_lim;
        int tk = idx - tb * k_lim;
        int gb = b0 + tb;
        int gk = k0 + tk;
        X0[tb * TK + tk] = (gb < B) ? X[(size_t)gb * K + gk] : 0.0f;
      }
    }
  }
  __syncthreads();

  // Main loop within this K-slice
  for (; k0 < k_end; k0 += TK, buf ^= 1){
    const int k_lim = dmin(TK, k_end - k0);
    float* x_cur = (buf == 0 ? X0 : X1);
    float* x_nxt = (buf == 0 ? X1 : X0);

    if (active){
      const float* __restrict__ wrow = W + (size_t)m * K + k0;
      const float* __restrict__ xrow = x_cur + (size_t)ty * TK;

      if (VEC4 && smem_aligned16 && W_base16 && stride16 && ((k0 & 3) == 0) && (k_lim >= 4)){
        const int vec_end = (k_lim >> 2) << 2;
        for (int jj = lane * 4; jj < vec_end; jj += 4 * WARP_SIZE){
          const Float4* __restrict__ w4 = reinterpret_cast<const Float4*>(wrow + jj);
          const Float4* __restrict__ x4 = reinterpret_cast<const Float4*>(xrow + jj);
          Float4 a = *w4;
          Float4 b = *x4;
          acc = fmaf(a.x, b.x, acc);
          acc = fmaf(a.y, b.y, acc);
          acc = fmaf(a.z, b.z, acc);
          acc = fmaf(a.w, b.w, acc);
        }
        for (int jj = ((k_lim>>2)<<2) + lane; jj < k_lim; jj += WARP_SIZE){
          acc = fmaf(wrow[jj], xrow[jj], acc);
        }
      } else {
        #pragma unroll 4
        for (int jj = lane; jj < k_lim; jj += WARP_SIZE){
          acc = fmaf(wrow[jj], xrow[jj], acc);
        }
      }
    }

    // Preload next X tile for this K-slice
    const int next_k0 = k0 + TK;
    if (next_k0 < k_end){
      const int next_lim = dmin(TK, k_end - next_k0);
      const bool vec4_ok = VEC4 && stride16 && ((next_k0 & 3) == 0) && (next_lim >= 4) && X_base16;

      if (vec4_ok){
        const int cols4  = (next_lim >> 2);
        const int total4 = TB * cols4;
        for (int idx4 = tidLinear; idx4 < total4; idx4 += threadsPerBlock){
          int tb  = idx4 / cols4;
          int tk4 = idx4 - tb * cols4;
          int gb  = b0 + tb;
          int gk  = next_k0 + (tk4 << 2);
          Float4 v = {0,0,0,0};
          if (gb < B){
            const Float4* px = reinterpret_cast<const Float4*>(
                X + (size_t)gb * K + gk);
            v = *px;
          }
          const int base = tb * TK + (tk4<<2);
          x_nxt[base + 0] = v.x;
          x_nxt[base + 1] = v.y;
          x_nxt[base + 2] = v.z;
          x_nxt[base + 3] = v.w;
        }
        const int vec_end = (cols4 << 2);
        const int tail    = next_lim - vec_end;
        if (tail){
          const int span2 = TB * tail;
          for (int idx = tidLinear; idx < span2; idx += threadsPerBlock){
            int tb = idx / tail;
            int tk = idx - tb * tail;
            int gb = b0 + tb;
            int gk = next_k0 + vec_end + tk;
            x_nxt[tb * TK + (vec_end + tk)] = (gb < B) ? X[(size_t)gb * K + gk] : 0.0f;
          }
        }
      } else {
        const int span2 = TB * next_lim;
        for (int idx = tidLinear; idx < span2; idx += threadsPerBlock){
          int tb = idx / next_lim;
          int tk = idx - tb * next_lim;
          int gb = b0 + tb;
          int gk = next_k0 + tk;
          x_nxt[tb * TK + tk] = (gb < B) ? X[(size_t)gb * K + gk] : 0.0f;
        }
      }
    }

    __syncthreads();
  }

  // Accumulate slice
  float sum = warp_reduce_sum(acc);
  if (lane == 0 && active){
    atomicAdd(&Y[(size_t)b * M + m], sum);
  }
}

// ============================== Launchers ==============================

template<int TK=1024, int TB=4, bool VEC4=true>
static inline void gemv_gpu_batch_opt(float* Y, const float* X, const float* W,
                                      int K, int M, int B,
                                      int waves_per_block = 4,
                                      hipStream_t stream = 0)
{
  // threads/block = TB * waves_per_block * WARP_SIZE  (<= 1024)
  const int threads_x = waves_per_block * WARP_SIZE;
  const int threads_y = TB;
  const int threads_per_block = threads_x * threads_y;
  // guard if user picks too large TB/waves
  if (threads_per_block > 1024){
    // fallback: shrink TB
    int newTB = 1024 / (waves_per_block * WARP_SIZE);
    if (newTB < 1) newTB = 1;
    printf("[gemv_gpu_batch_opt] Adjusting TB from %d to %d to satisfy 1024-thread limit.\n", TB, newTB);
  }

  dim3 block(threads_x, threads_y, 1);
  dim3 grid((M + waves_per_block - 1) / waves_per_block,
            (B + TB - 1) / TB,
            1);
  size_t shmem = 2ULL * TB * TK * sizeof(float);

  // No split-K: kernel overwrites Y directly (caller should not pre-zero)
  hipLaunchKernelGGL((k_gemv_batched_wave_row_shared<TK,TB,VEC4>),
                     grid, block, shmem, stream,
                     W, X, Y, K, M, B);
}

template<int TK=1024, int TB=4, bool VEC4=true>
static inline void gemv_gpu_batch_opt_splitK(float* Y, const float* X, const float* W,
                                             int K, int M, int B,
                                             int splitK,
                                             int waves_per_block = 4,
                                             hipStream_t stream = 0)
{
  const int threads_x = waves_per_block * WARP_SIZE;
  const int threads_y = TB;
  dim3 block(threads_x, threads_y, 1);
  dim3 grid((M + waves_per_block - 1) / waves_per_block,
            (B + TB - 1) / TB,
            splitK);
  size_t shmem = 2ULL * TB * TK * sizeof(float);

  // split-K uses atomicAdd → cần zero trước
  hipMemsetAsync(Y, 0, (size_t)B * M * sizeof(float), stream);

  hipLaunchKernelGGL((k_gemv_batched_wave_row_shared_splitK<TK,TB,VEC4>),
                     grid, block, shmem, stream,
                     W, X, Y, K, M, B, splitK);
}

// ============================== Heuristic wrapper ==============================
// Chọn giữa no-splitK và splitK; thiết lập thông số an toàn cho MI2x0/MI3x0.
static inline void gemv_gpu_batch(float* Y, const float* X, const float* W,
                                  int K, int M, int B,
                                  hipStream_t stream = 0)
{
  // Heuristic:
  // - TK=2048 tăng hiệu năng khi K lớn nhưng ngốn LDS; với TB=4 → 2*4*2048*4 = 64 KiB (OK).
  // - waves_per_block=4 (256 threads trên mỗi hàng sample) cân bằng latency/occupancy cho gfx90a/gfx94.
  constexpr int TK = 2048;
  constexpr int TB = 4;
  const int waves_per_block = 4;

  int splitK = 1;
  if      (K >= 65536) splitK = 8;
  else if (K >= 32768) splitK = 4;
  else if (K >= 16384) splitK = 2;

  if (splitK == 1){
    gemv_gpu_batch_opt<TK,TB,true>(Y, X, W, K, M, B, waves_per_block, stream);
  } else {
    gemv_gpu_batch_opt_splitK<TK,TB,true>(Y, X, W, K, M, B, splitK, waves_per_block, stream);
  }
}

__device__ __forceinline__ float bf16_to_float(hip_bfloat16 h) {
  // hip_bfloat16 là 16-bit (giống chuẩn IEEE754 BF16: 8-bit exponent, 7-bit mantissa)
  // Chuyển bằng cách nhét 16 bit lên high bits của float32
  uint16_t lo = *reinterpret_cast<const uint16_t*>(&h);
  uint32_t hi = (uint32_t)lo << 16;
  union { uint32_t u; float f; } cvt;
  cvt.u = hi;
  return cvt.f;
}

// ================= Wave-per-row batched (no split-K) =================
template<int TK=1024, int TB=4, bool VEC4=true>
__global__ void k_gemv_batched_wave_row_shared_bf16(const hip_bfloat16* __restrict__ W, // [M,K] (BF16)
                                               const float*       __restrict__ X, // [B,K]
                                               float*             __restrict__ Y, // [B,M]
                                               int K, int M, int B)
{
  extern __shared__ float s[];
  float* X0 = s;
  float* X1 = s + (size_t)TB * TK;

  const int wid  = warp_id();        // 0..waves_per_block-1
  const int lane = lane_id();        // 0..WARP_SIZE-1
  const int waves_per_block = blockDim.x / WARP_SIZE;

  const int b0 = blockIdx.y * TB;
  const int m0 = blockIdx.x * waves_per_block;

  const int ty = threadIdx.y;        // 0..TB-1 (sample lane)
  const int b  = b0 + ty;            // global sample
  const int m  = m0 + wid;           // global row (per-wave)

  const bool active = (b < B) && (m < M);
  const bool smem_aligned16 = (((uintptr_t)s & 0xF) == 0);
  const bool stride16 = ((K & 3) == 0);
  const bool W_base16 = (((uintptr_t)W & 0xF) == 0);
  const bool X_base16 = (((uintptr_t)X & 0xF) == 0);

  const int threadsPerBlock = blockDim.x * blockDim.y;
  const int tidLinear = threadIdx.y * blockDim.x + threadIdx.x;

  float acc = 0.0f;
  int   buf = 0;

  // ---- Preload first tile of X into X0 ----
  int k0 = 0;
  {
    const int k_lim = dmin(TK, K - k0);
    const bool vec4_ok = VEC4 && stride16 && ((k0 & 3) == 0) && (k_lim >= 4) && X_base16;

    if (vec4_ok){
      const int cols4  = (k_lim >> 2);
      const int total4 = TB * cols4;
      for (int idx4 = tidLinear; idx4 < total4; idx4 += threadsPerBlock){
        int tb  = idx4 / cols4;
        int tk4 = idx4 - tb * cols4;
        int gb  = b0 + tb;
        int gk  = k0 + (tk4 << 2);
        Float4 v = {0,0,0,0};
        if (gb < B){
          const Float4* px = reinterpret_cast<const Float4*>(X + (size_t)gb * K + gk);
          v = *px;
        }
        const int base = tb * TK + (tk4<<2);
        if (0 <  k_lim) X0[base + 0] = v.x;
        if (1 <  k_lim) X0[base + 1] = v.y;
        if (2 <  k_lim) X0[base + 2] = v.z;
        if (3 <  k_lim) X0[base + 3] = v.w;
      }
      // tail
      const int vec_end = (cols4 << 2);
      const int tail    = k_lim - vec_end;
      if (tail){
        const int span = TB * tail;
        for (int idx = tidLinear; idx < span; idx += threadsPerBlock){
          int tb = idx / tail;
          int tk = idx - tb * tail;
          int gb = b0 + tb;
          int gk = k0 + vec_end + tk;
          X0[tb * TK + (vec_end + tk)] = (gb < B) ? X[(size_t)gb * K + gk] : 0.0f;
        }
      }
    } else {
      const int span = TB * k_lim;
      for (int idx = tidLinear; idx < span; idx += threadsPerBlock){
        int tb = idx / k_lim;
        int tk = idx - tb * k_lim;
        int gb = b0 + tb;
        int gk = k0 + tk;
        X0[tb * TK + tk] = (gb < B) ? X[(size_t)gb * K + gk] : 0.0f;
      }
    }
  }
  __syncthreads();

  // ---- Main K loop (ping-pong X) ----
  for (; k0 < K; k0 += TK, buf ^= 1){
    const int k_lim = dmin(TK, K - k0);
    float* x_cur = (buf == 0 ? X0 : X1);
    float* x_nxt = (buf == 0 ? X1 : X0);

    if (active){
      const hip_bfloat16* __restrict__ wrow = W + (size_t)m * K + k0;
      const float*        __restrict__ xrow = x_cur + (size_t)ty * TK;

      if (VEC4 && smem_aligned16 && W_base16 && stride16 && ((k0 & 3) == 0) && (k_lim >= 4)){
        const int vec_end = (k_lim >> 2) << 2;
        for (int jj = lane * 4; jj < vec_end; jj += 4 * WARP_SIZE){
          // Vector read X (shared), W đọc 4 phần tử bfloat16 rồi convert
          const Float4* __restrict__ x4 = reinterpret_cast<const Float4*>(xrow + jj);
          Float4 b = *x4;
          float a0 = bf16_to_float(wrow[jj+0]);
          float a1 = bf16_to_float(wrow[jj+1]);
          float a2 = bf16_to_float(wrow[jj+2]);
          float a3 = bf16_to_float(wrow[jj+3]);
          acc = fmaf(a0, b.x, acc);
          acc = fmaf(a1, b.y, acc);
          acc = fmaf(a2, b.z, acc);
          acc = fmaf(a3, b.w, acc);
        }
        for (int jj = ((k_lim>>2)<<2) + lane; jj < k_lim; jj += WARP_SIZE){
          acc = fmaf(bf16_to_float(wrow[jj]), xrow[jj], acc);
        }
      } else {
        #pragma unroll 4
        for (int jj = lane; jj < k_lim; jj += WARP_SIZE){
          acc = fmaf(bf16_to_float(wrow[jj]), xrow[jj], acc);
        }
      }
    }

    // ---- Preload next X tile (cooperative) ----
    const int next_k0 = k0 + TK;
    if (next_k0 < K){
      const int next_lim = dmin(TK, K - next_k0);
      const bool vec4_ok = VEC4 && stride16 && ((next_k0 & 3) == 0) && (next_lim >= 4) && X_base16;

      if (vec4_ok){
        const int cols4  = (next_lim >> 2);
        const int total4 = TB * cols4;
        for (int idx4 = tidLinear; idx4 < total4; idx4 += threadsPerBlock){
          int tb  = idx4 / cols4;
          int tk4 = idx4 - tb * cols4;
          int gb  = b0 + tb;
          int gk  = next_k0 + (tk4 << 2);
          Float4 v = {0,0,0,0};
          if (gb < B){
            const Float4* px = reinterpret_cast<const Float4*>(X + (size_t)gb * K + gk);
            v = *px;
          }
          const int base = tb * TK + (tk4<<2);
          if (0 < next_lim) x_nxt[base + 0] = v.x;
          if (1 < next_lim) x_nxt[base + 1] = v.y;
          if (2 < next_lim) x_nxt[base + 2] = v.z;
          if (3 < next_lim) x_nxt[base + 3] = v.w;
        }
        const int vec_end = (cols4 << 2);
        const int tail    = next_lim - vec_end;
        if (tail){
          const int span = TB * tail;
          for (int idx = tidLinear; idx < span; idx += threadsPerBlock){
            int tb = idx / tail;
            int tk = idx - tb * tail;
            int gb = b0 + tb;
            int gk = next_k0 + vec_end + tk;
            x_nxt[tb * TK + (vec_end + tk)] = (gb < B) ? X[(size_t)gb * K + gk] : 0.0f;
          }
        }
      } else {
        const int span = TB * next_lim;
        for (int idx = tidLinear; idx < span; idx += threadsPerBlock){
          int tb = idx / next_lim;
          int tk = idx - tb * next_lim;
          int gb = b0 + tb;
          int gk = next_k0 + tk;
          x_nxt[tb * TK + tk] = (gb < B) ? X[(size_t)gb * K + gk] : 0.0f;
        }
      }
    }

    __syncthreads();
  }

  // ---- Write result ----
  float sum = warp_reduce_sum(acc);
  if (lane == 0 && active){
    Y[(size_t)b * M + m] = sum;
  }
}

// ================= Split-K batched (atomicAdd to Y) =================
template<int TK=1024, int TB=4, bool VEC4=true>
__global__ void k_gemv_batched_wave_row_shared_splitK_bf16(const hip_bfloat16* __restrict__ W, // BF16
                                                      const float*       __restrict__ X,
                                                      float*             __restrict__ Y, // atomicAdd
                                                      int K, int M, int B, int splitK)
{
  extern __shared__ float s[];
  float* X0 = s;
  float* X1 = s + (size_t)TB * TK;

  const int wid  = warp_id();
  const int lane = lane_id();
  const int waves_per_block = blockDim.x / WARP_SIZE;

  const int b0 = blockIdx.y * TB;
  const int m0 = blockIdx.x * waves_per_block;
  const int part = blockIdx.z;

  const int span    = dceil_div(K, splitK);
  const int k_begin = part * span;
  const int k_end   = dmin(K, k_begin + span);
  if (k_begin >= k_end) return;

  const int ty = threadIdx.y;
  const int b  = b0 + ty;
  const int m  = m0 + wid;

  const bool active = (b < B) && (m < M);
  const bool smem_aligned16 = (((uintptr_t)s & 0xF) == 0);
  const bool stride16 = ((K & 3) == 0);
  const bool W_base16 = (((uintptr_t)W & 0xF) == 0);
  const bool X_base16 = (((uintptr_t)X & 0xF) == 0);

  const int threadsPerBlock = blockDim.x * blockDim.y;
  const int tidLinear = threadIdx.y * blockDim.x + threadIdx.x;

  float acc = 0.0f;
  int   buf = 0;

  // Preload first X tile
  int k0 = k_begin;
  {
    const int k_lim = dmin(TK, k_end - k0);
    const bool vec4_ok = VEC4 && stride16 && ((k0 & 3) == 0) && (k_lim >= 4) && X_base16;

    if (vec4_ok){
      const int cols4  = (k_lim >> 2);
      const int total4 = TB * cols4;
      for (int idx4 = tidLinear; idx4 < total4; idx4 += threadsPerBlock){
        int tb  = idx4 / cols4;
        int tk4 = idx4 - tb * cols4;
        int gb  = b0 + tb;
        int gk  = k0 + (tk4 << 2);
        Float4 v = {0,0,0,0};
        if (gb < B){
          const Float4* px = reinterpret_cast<const Float4*>(X + (size_t)gb * K + gk);
          v = *px;
        }
        const int base = tb * TK + (tk4<<2);
        if (0 <  k_lim) X0[base + 0] = v.x;
        if (1 <  k_lim) X0[base + 1] = v.y;
        if (2 <  k_lim) X0[base + 2] = v.z;
        if (3 <  k_lim) X0[base + 3] = v.w;
      }
      const int vec_end = (cols4 << 2);
      const int tail    = k_lim - vec_end;
      if (tail){
        const int span2 = TB * tail;
        for (int idx = tidLinear; idx < span2; idx += threadsPerBlock){
          int tb = idx / tail;
          int tk = idx - tb * tail;
          int gb = b0 + tb;
          int gk = k0 + vec_end + tk;
          X0[tb * TK + (vec_end + tk)] = (gb < B) ? X[(size_t)gb * K + gk] : 0.0f;
        }
      }
    } else {
      const int span2 = TB * k_lim;
      for (int idx = tidLinear; idx < span2; idx += threadsPerBlock){
        int tb = idx / k_lim;
        int tk = idx - tb * k_lim;
        int gb = b0 + tb;
        int gk = k0 + tk;
        X0[tb * TK + tk] = (gb < B) ? X[(size_t)gb * K + gk] : 0.0f;
      }
    }
  }
  __syncthreads();

  // Main loop within this K-slice
  for (; k0 < k_end; k0 += TK, buf ^= 1){
    const int k_lim = dmin(TK, k_end - k0);
    float* x_cur = (buf == 0 ? X0 : X1);
    float* x_nxt = (buf == 0 ? X1 : X0);

    if (active){
      const hip_bfloat16* __restrict__ wrow = W + (size_t)m * K + k0;
      const float*        __restrict__ xrow = x_cur + (size_t)ty * TK;

      if (VEC4 && smem_aligned16 && W_base16 && stride16 && ((k0 & 3) == 0) && (k_lim >= 4)){
        const int vec_end = (k_lim >> 2) << 2;
        for (int jj = lane * 4; jj < vec_end; jj += 4 * WARP_SIZE){
          const Float4* __restrict__ x4 = reinterpret_cast<const Float4*>(xrow + jj);
          Float4 b = *x4;
          float a0 = bf16_to_float(wrow[jj+0]);
          float a1 = bf16_to_float(wrow[jj+1]);
          float a2 = bf16_to_float(wrow[jj+2]);
          float a3 = bf16_to_float(wrow[jj+3]);
          acc = fmaf(a0, b.x, acc);
          acc = fmaf(a1, b.y, acc);
          acc = fmaf(a2, b.z, acc);
          acc = fmaf(a3, b.w, acc);
        }
        for (int jj = ((k_lim>>2)<<2) + lane; jj < k_lim; jj += WARP_SIZE){
          acc = fmaf(bf16_to_float(wrow[jj]), xrow[jj], acc);
        }
      } else {
        #pragma unroll 4
        for (int jj = lane; jj < k_lim; jj += WARP_SIZE){
          acc = fmaf(bf16_to_float(wrow[jj]), xrow[jj], acc);
        }
      }
    }

    // Preload next X tile for this K-slice
    const int next_k0 = k0 + TK;
    if (next_k0 < k_end){
      const int next_lim = dmin(TK, k_end - next_k0);
      const bool vec4_ok = VEC4 && stride16 && ((next_k0 & 3) == 0) && (next_lim >= 4) && X_base16;

      if (vec4_ok){
        const int cols4  = (next_lim >> 2);
        const int total4 = TB * cols4;
        for (int idx4 = tidLinear; idx4 < total4; idx4 += threadsPerBlock){
          int tb  = idx4 / cols4;
          int tk4 = idx4 - tb * cols4;
          int gb  = b0 + tb;
          int gk  = next_k0 + (tk4 << 2);
          Float4 v = {0,0,0,0};
          if (gb < B){
            const Float4* px = reinterpret_cast<const Float4*>(X + (size_t)gb * K + gk);
            v = *px;
          }
          const int base = tb * TK + (tk4<<2);
          x_nxt[base + 0] = v.x;
          x_nxt[base + 1] = v.y;
          x_nxt[base + 2] = v.z;
          x_nxt[base + 3] = v.w;
        }
        const int vec_end = (cols4 << 2);
        const int tail    = next_lim - vec_end;
        if (tail){
          const int span2 = TB * tail;
          for (int idx = tidLinear; idx < span2; idx += threadsPerBlock){
            int tb = idx / tail;
            int tk = idx - tb * tail;
            int gb = b0 + tb;
            int gk = next_k0 + vec_end + tk;
            x_nxt[tb * TK + (vec_end + tk)] = (gb < B) ? X[(size_t)gb * K + gk] : 0.0f;
          }
        }
      } else {
        const int span2 = TB * next_lim;
        for (int idx = tidLinear; idx < span2; idx += threadsPerBlock){
          int tb = idx / next_lim;
          int tk = idx - tb * next_lim;
          int gb = b0 + tb;
          int gk = next_k0 + tk;
          x_nxt[tb * TK + tk] = (gb < B) ? X[(size_t)gb * K + gk] : 0.0f;
        }
      }
    }

    __syncthreads();
  }

  // Accumulate slice
  float sum = warp_reduce_sum(acc);
  if (lane == 0 && active){
    atomicAdd(&Y[(size_t)b * M + m], sum);
  }
}

// ============================== Launchers ==============================

template<int TK=1024, int TB=4, bool VEC4=true>
static inline void gemv_gpu_batch_opt_bf16(float* Y, const float* X, const hip_bfloat16* W,
                                      int K, int M, int B,
                                      int waves_per_block = 4,
                                      hipStream_t stream = 0)
{
  const int threads_x = waves_per_block * WARP_SIZE;
  const int threads_y = TB;
  const int threads_per_block = threads_x * threads_y;
  if (threads_per_block > 1024){
    int newTB = 1024 / (waves_per_block * WARP_SIZE);
    if (newTB < 1) newTB = 1;
    printf("[gemv_gpu_batch_opt] Adjusting TB from %d to %d to satisfy 1024-thread limit.\n", TB, newTB);
  }

  dim3 block(threads_x, threads_y, 1);
  dim3 grid((M + waves_per_block - 1) / waves_per_block,
            (B + TB - 1) / TB,
            1);
  size_t shmem = 2ULL * TB * TK * sizeof(float);

  hipLaunchKernelGGL((k_gemv_batched_wave_row_shared_bf16<TK,TB,VEC4>),
                     grid, block, shmem, stream,
                     W, X, Y, K, M, B);
}

template<int TK=1024, int TB=4, bool VEC4=true>
static inline void gemv_gpu_batch_opt_splitK_bf16(float* Y, const float* X, const hip_bfloat16* W,
                                             int K, int M, int B,
                                             int splitK,
                                             int waves_per_block = 4,
                                             hipStream_t stream = 0)
{
  const int threads_x = waves_per_block * WARP_SIZE;
  const int threads_y = TB;
  dim3 block(threads_x, threads_y, 1);
  dim3 grid((M + waves_per_block - 1) / waves_per_block,
            (B + TB - 1) / TB,
            splitK);
  size_t shmem = 2ULL * TB * TK * sizeof(float);

  hipMemsetAsync(Y, 0, (size_t)B * M * sizeof(float), stream);

  hipLaunchKernelGGL((k_gemv_batched_wave_row_shared_splitK_bf16<TK,TB,VEC4>),
                     grid, block, shmem, stream,
                     W, X, Y, K, M, B, splitK);
}

// ============================== Heuristic wrapper ==============================
static inline void gemv_gpu_batch_bf16(float* Y, const float* X, const hip_bfloat16* W,
                                  int K, int M, int B,
                                  hipStream_t stream = 0)
{
  constexpr int TK = 2048;
  constexpr int TB = 4;
  const int waves_per_block = 4;

  int splitK = 1;
  if      (K >= 65536) splitK = 8;
  else if (K >= 32768) splitK = 4;
  else if (K >= 16384) splitK = 2;

  if (splitK == 1){
    gemv_gpu_batch_opt_bf16<TK,TB,true>(Y, X, W, K, M, B, waves_per_block, stream);
  } else {
    gemv_gpu_batch_opt_splitK_bf16<TK,TB,true>(Y, X, W, K, M, B, splitK, waves_per_block, stream);
  }
}