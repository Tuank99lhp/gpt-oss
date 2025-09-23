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
// Scalar BF16 -> F32 (fallback)
__device__ __forceinline__ float bf16_to_float(hip_bfloat16 h) {
  uint16_t lo = *reinterpret_cast<const uint16_t*>(&h);
  uint32_t hi = (uint32_t)lo << 16;
  return __uint_as_float(hi);
}

// Packed BF16 loaders (aligned paths)
// Load 2x bf16 as one u32 -> two f32
__device__ __forceinline__ Float2 ld_bf16x2_to_f32x2(const hip_bfloat16* __restrict__ p) {
  // Using a plain 32-bit load. Base alignment checks guard hot paths.
  uint32_t u = *reinterpret_cast<const uint32_t*>(p);
  Float2 r;
  r.x = __uint_as_float((u & 0x0000FFFFu) << 16);
  r.y = __uint_as_float( u & 0xFFFF0000u);
  return r;
}

// Load 4x bf16 as one u64 -> four f32
__device__ __forceinline__ Float4 ld_bf16x4_to_f32x4(const hip_bfloat16* __restrict__ p) {
  uint64_t u = *reinterpret_cast<const uint64_t*>(p);
  uint32_t lo = static_cast<uint32_t>(u);
  uint32_t hi = static_cast<uint32_t>(u >> 32);

  Float4 r;
  // order: b0 (lo low16), b1 (lo high16), b2 (hi low16), b3 (hi high16)
  r.x = __uint_as_float((lo & 0x0000FFFFu) << 16);
  r.y = __uint_as_float( lo & 0xFFFF0000u);
  r.z = __uint_as_float((hi & 0x0000FFFFu) << 16);
  r.w = __uint_as_float( hi & 0xFFFF0000u);
  return r;
}

// --------------------- Cooperative LDS loaders for X ---------------------
template<bool VEC4>
__device__ __forceinline__
void preload_X_tile(float* __restrict__ dst_tile,        // [TB,TK] in LDS
                    const float* __restrict__ X,         // [B,K]
                    int b0, int TK, int K, int k0, int B,
                    int TB, int threadsPerBlock, int tidLinear,
                    bool X_base16)
{
  const int k_lim = dmin(TK, K - k0);
  const bool stride16 = ((K & 3) == 0);
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
      // bounds already guaranteed by k_lim
      dst_tile[base + 0] = v.x;
      dst_tile[base + 1] = v.y;
      dst_tile[base + 2] = v.z;
      dst_tile[base + 3] = v.w;
    }

    // tiny tail (<=3)
    const int vec_end = (cols4 << 2);
    const int tail    = k_lim - vec_end;
    if (tail){
      const int span = TB * tail;
      for (int idx = tidLinear; idx < span; idx += threadsPerBlock){
        int tb = idx / tail;
        int tk = idx - tb * tail;
        int gb = b0 + tb;
        int gk = k0 + vec_end + tk;
        dst_tile[tb * TK + (vec_end + tk)] = (gb < B) ? X[(size_t)gb * K + gk] : 0.0f;
      }
    }
  } else {
    const int span = TB * k_lim;
    for (int idx = tidLinear; idx < span; idx += threadsPerBlock){
      int tb = idx / k_lim;
      int tk = idx - tb * k_lim;
      int gb = b0 + tb;
      int gk = k0 + tk;
      dst_tile[tb * TK + tk] = (gb < B) ? X[(size_t)gb * K + gk] : 0.0f;
    }
  }
}

// ================= Wave-per-row batched (no split-K) =================
template<int TK=1024, int TB=4, bool VEC4=true>
__global__ void k_gemv_batched_wave_row_shared_bf16(
    const hip_bfloat16* __restrict__ W, // [M,K] (BF16)
    const float*       __restrict__ X,  // [B,K] (F32)
    float*             __restrict__ Y,  // [B,M] (F32)
    int K, int M, int B)
{
  extern __shared__ float s[];
  float* X0 = s;
  float* X1 = s + (size_t)TB * TK;

  const int wid  = warp_id();                 // 0..waves_per_block-1
  const int lane = lane_id();                 // 0..WARP_SIZE-1
  const int waves_per_block = blockDim.x / WARP_SIZE;

  const int b0 = blockIdx.y * TB;
  const int m0 = blockIdx.x * waves_per_block;

  const int ty = threadIdx.y;                 // 0..TB-1 (sample lane)
  const int b  = b0 + ty;                     // global sample
  const int m  = m0 + wid;                    // global row (per-wave)

  const bool active = (b < B) && (m < M);
  const bool smem_aligned16 = (((uintptr_t)s & 0xF) == 0);
  const bool stride16 = ((K & 3) == 0);
  const bool W_base16 = (((uintptr_t)W & 0xF) == 0);
  const bool X_base16 = (((uintptr_t)X & 0xF) == 0);

  const int threadsPerBlock = blockDim.x * blockDim.y;
  const int tidLinear = threadIdx.y * blockDim.x + threadIdx.x;

  float acc = 0.0f;
  int   buf = 0;

  // ---- Preload first X tile into X0 ----
  int k0 = 0;
  preload_X_tile<VEC4>((float*)X0, X, b0, TK, K, k0, B, TB, threadsPerBlock, tidLinear, X_base16);
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

        #pragma unroll 2
        for (int jj = lane * 4; jj < vec_end; jj += 4 * WARP_SIZE){
          // Vectorized loads: X from LDS, W as bf16x4 -> f32x4
          const Float4* __restrict__ x4p = reinterpret_cast<const Float4*>(xrow + jj);
          Float4 xb = *x4p;
          Float4 aw = ld_bf16x4_to_f32x4(wrow + jj);

          acc = fmaf(aw.x, xb.x, acc);
          acc = fmaf(aw.y, xb.y, acc);
          acc = fmaf(aw.z, xb.z, acc);
          acc = fmaf(aw.w, xb.w, acc);
        }
        // tail
        for (int jj = vec_end + lane; jj < k_lim; jj += WARP_SIZE){
          acc = fmaf(bf16_to_float(wrow[jj]), xrow[jj], acc);
        }
      } else {
        // scalar fallback
        #pragma unroll 4
        for (int jj = lane; jj < k_lim; jj += WARP_SIZE){
          acc = fmaf(bf16_to_float(wrow[jj]), xrow[jj], acc);
        }
      }
    }

    // ---- Preload next X tile (cooperative) ----
    const int next_k0 = k0 + TK;
    if (next_k0 < K){
      preload_X_tile<VEC4>((float*)x_nxt, X, b0, TK, K, next_k0, B, TB, threadsPerBlock, tidLinear, X_base16);
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
__global__ void k_gemv_batched_wave_row_shared_splitK_bf16(
    const hip_bfloat16* __restrict__ W, // [M,K] (BF16)
    const float*       __restrict__ X,  // [B,K] (F32)
    float*             __restrict__ Y,  // [B,M] (F32) atomicAdd
    int K, int M, int B, int splitK)
{
  extern __shared__ float s[];
  float* X0 = s;
  float* X1 = s + (size_t)TB * TK;

  const int wid  = warp_id();
  const int lane = lane_id();
  const int waves_per_block = blockDim.x / WARP_SIZE;

  const int b0   = blockIdx.y * TB;
  const int m0   = blockIdx.x * waves_per_block;
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
  preload_X_tile<VEC4>((float*)X0, X, b0, TK, K, k0, B, TB, threadsPerBlock, tidLinear, X_base16);
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

        #pragma unroll 2
        for (int jj = lane * 4; jj < vec_end; jj += 4 * WARP_SIZE){
          const Float4* __restrict__ x4p = reinterpret_cast<const Float4*>(xrow + jj);
          Float4 xb = *x4p;
          Float4 aw = ld_bf16x4_to_f32x4(wrow + jj);

          acc = fmaf(aw.x, xb.x, acc);
          acc = fmaf(aw.y, xb.y, acc);
          acc = fmaf(aw.z, xb.z, acc);
          acc = fmaf(aw.w, xb.w, acc);
        }
        // tail
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
      preload_X_tile<VEC4>((float*)x_nxt, X, b0, TK, K, next_k0, B, TB, threadsPerBlock, tidLinear, X_base16);
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
static inline void gemv_gpu_batch_opt_bf16(
    float* Y, const float* X, const hip_bfloat16* W,
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
    fprintf(stderr,
      "[gemv_gpu_batch_opt_bf16] Warning: block size %d exceeds 1024. "
      "Recompile with smaller TB or fewer waves_per_block (TB=%d, waves=%d)\n",
      threads_per_block, newTB, waves_per_block);
  }

  dim3 block(threads_x, threads_y, 1);
  dim3 grid(ceil_div(M, waves_per_block),
            ceil_div(B, TB),
            1);
  size_t shmem = 2ULL * TB * TK * sizeof(float);

  hipLaunchKernelGGL((k_gemv_batched_wave_row_shared_bf16<TK,TB,VEC4>),
                     grid, block, shmem, stream,
                     W, X, Y, K, M, B);
}

template<int TK=1024, int TB=4, bool VEC4=true>
static inline void gemv_gpu_batch_opt_splitK_bf16(
    float* Y, const float* X, const hip_bfloat16* W,
    int K, int M, int B,
    int splitK,
    int waves_per_block = 4,
    hipStream_t stream = 0)
{
  const int threads_x = waves_per_block * WARP_SIZE;
  const int threads_y = TB;
  dim3 block(threads_x, threads_y, 1);
  dim3 grid(ceil_div(M, waves_per_block),
            ceil_div(B, TB),
            splitK);
  size_t shmem = 2ULL * TB * TK * sizeof(float);

  HIP_CHECK( hipMemsetAsync(Y, 0, (size_t)B * M * sizeof(float), stream) );

  hipLaunchKernelGGL((k_gemv_batched_wave_row_shared_splitK_bf16<TK,TB,VEC4>),
                     grid, block, shmem, stream,
                     W, X, Y, K, M, B, splitK);
}

// ============================== Heuristic wrapper ==============================
static inline void gemv_gpu_batch_bf16(
    float* Y, const float* X, const hip_bfloat16* W,
    int K, int M, int B,
    hipStream_t stream = 0)
{
  // Prefer large TK for bandwidth efficiency; keep multiple of 32.
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