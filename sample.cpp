// ---------------- Warp helpers ----------------
__device__ __forceinline__ float shfl_down_f(float v, int off) {
  return __shfl_down(v, off, WARP_SIZE);
}
__device__ __forceinline__ int shfl_down_i(int v, int off) {
  return __shfl_down(v, off, WARP_SIZE);
}

// Reduce (max,value) kèm index với tie-break: nếu bằng nhau -> chọn index nhỏ hơn
__device__ __forceinline__ void warp_argmax_reduce(float &val, int &idx) {
  for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) {
    float v2 = shfl_down_f(val, off);
    int   i2 = shfl_down_i(idx, off);
    if (v2 > val || (v2 == val && i2 < idx)) {
      val = v2; idx = i2;
    }
  }
}

__global__ void k_argmax_rows_opt(const float* __restrict__ logits,
                                  int V, int B, int use_vec4,
                                  int* __restrict__ out_idx) {
  const int b = blockIdx.x;
  if (b >= B) return;

  const float* row = logits + (size_t)b * (size_t)V;

  // 1) Mỗi thread quét phần được phân công (grid-stride theo block) và giữ local (max,idx)
  float local_max = -INFINITY;
  int   local_idx = 0;

  if (use_vec4) {
    // vectorized path: V % 4 == 0 và row 16B aligned
    const size_t V4 = (size_t)V >> 2;
    const float4* __restrict__ p4 = reinterpret_cast<const float4*>(row);
    for (size_t i4 = threadIdx.x; i4 < V4; i4 += blockDim.x) {
      float4 v4 = p4[i4];
      int base = (int)(i4 << 2);
      // so sánh từng lane của float4
      // 0
      if (v4.x > local_max || (v4.x == local_max && base + 0 < local_idx)) {
        local_max = v4.x; local_idx = base + 0;
      }
      // 1
      if (v4.y > local_max || (v4.y == local_max && base + 1 < local_idx)) {
        local_max = v4.y; local_idx = base + 1;
      }
      // 2
      if (v4.z > local_max || (v4.z == local_max && base + 2 < local_idx)) {
        local_max = v4.z; local_idx = base + 2;
      }
      // 3
      if (v4.w > local_max || (v4.w == local_max && base + 3 < local_idx)) {
        local_max = v4.w; local_idx = base + 3;
      }
    }
  } else {
    // scalar path
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
      float v = row[j];
      if (v > local_max || (v == local_max && j < local_idx)) {
        local_max = v; local_idx = j;
      }
    }
  }

  // 2) Warp-level reduce (max,idx)
  warp_argmax_reduce(local_max, local_idx);

  // 3) Block-level reduce qua shared (mỗi warp 1 kết quả)
  __shared__ float s_val[32];
  __shared__ int   s_idx[32];
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;
  const int nwarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

  if (lane == 0) { s_val[warp] = local_max; s_idx[warp] = local_idx; }
  __syncthreads();

  if (warp == 0) {
    float v = (lane < nwarps) ? s_val[lane] : -INFINITY;
    int   i = (lane < nwarps) ? s_idx[lane] : 0;
    warp_argmax_reduce(v, i);
    if (lane == 0) out_idx[b] = i;
  }
}

// ---------------- Launcher ----------------
static inline void sample_argmax_gpu_batch(const float* logits_d,
                                           int* out_idx_d,
                                           int B, int V,
                                           hipStream_t stream = 0) {
  if (B <= 0 || V <= 0) return;

  // Chọn cấu hình block hợp lý (bội số WARP_SIZE)
  const int BLOCK = 256; // 4 waves trên AMD
  dim3 grid(B), block(BLOCK);

  // Quyết định vector hóa: cần base ptr căn 16 byte và V % 4 == 0
  // Nếu base 16B aligned và stride theo hàng là V*4 bytes -> mọi hàng đều giữ căn 16B khi V%4==0
  const uintptr_t base = reinterpret_cast<uintptr_t>(logits_d);
  const int use_vec4 = ((base & 0xF) == 0 && (V % 4) == 0) ? 1 : 0;

  hipLaunchKernelGGL(k_argmax_rows_opt, grid, block, 0, stream,
                     logits_d, V, B, use_vec4, out_idx_d);
  // (khuyến nghị) kiểm tra lỗi khi debug:
  // hipError_t err = hipGetLastError(); if (err != hipSuccess) fprintf(stderr, "HIP: %s\n", hipGetErrorString(err));
}

__global__ void k_scale_logits(float* __restrict__ logits, long long n, float invT) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) logits[i] *= invT;
}

static inline void scale_logits(float* logits_d, float temperature,
                                int B, int V, hipStream_t stream = 0) {
  if (temperature <= 0.0f) return; // trường hợp argmax, không scale
  float invT = 1.0f / temperature;
  long long n = 1ll * B * V;
  const int BS = 256, GS = (n + BS - 1) / BS;
  hipLaunchKernelGGL(k_scale_logits, dim3(GS), dim3(BS), 0, stream, logits_d, n, invT);
}

static inline void get_coins_host(Sampler* sampler, float* coins, int B) {
  for (int i = 0; i < B; ++i) {
    coins[i] = random_f32(&sampler->rng_state); // UPDATE rng_state như CPU
  }
}

__global__ void k_sample_mult_rows(const float* __restrict__ probs, int V, int B,
                                   const float* __restrict__ coins,
                                   int* __restrict__ out_idx) {
  int b = blockIdx.x;
  if (b >= B) return;

  const float* row = probs + (size_t)b * V;
  float coin = coins[b];
  float cdf = 0.f;
  for (int i = 0; i < V; ++i) {
    cdf += row[i];
    if (coin < cdf) { out_idx[b] = i; return; }
  }
  out_idx[b] = V - 1; // dự phòng
}

static inline void sample_mult_gpu_batch(const float* probs_d, int B, int V,
                                         const float* d_coins, int* out_idx_d,
                                         hipStream_t stream = 0) {
  dim3 grid(B), block(1);
  hipLaunchKernelGGL(k_sample_mult_rows, grid, block, 0, stream,
                     probs_d, V, B, d_coins, out_idx_d);
}

// ===== 4) Top-p (nucleus) sampling: dùng buffer d_probindex [B*V] =====
__device__ void insertion_sort_desc(ProbIndex* arr, int n) {
  for (int i = 1; i < n; ++i) {
    ProbIndex key = arr[i];
    int j = i - 1;
    while (j >= 0 && arr[j].prob < key.prob) {
      arr[j + 1] = arr[j];
      --j;
    }
    arr[j + 1] = key;
  }
}

__device__ float row_sum_geq(const float* __restrict__ row, int V,
                             float tau, float* __restrict__ redbuf) {
  const int tid    = threadIdx.x;
  const int lane   = tid & (WARP_SIZE - 1);
  const int warp   = tid / WARP_SIZE;
  const int nwarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

  float local = 0.f;
  for (int i = tid; i < V; i += blockDim.x) {
    float p = row[i];
    if (p >= tau) local += p;
  }
  float wsum = warp_reduce_sum(local);
  if (lane == 0) redbuf[warp] = wsum;
  __syncthreads();

  float sum = 0.f;
  if (warp == 0) {
    float v = (lane < nwarps) ? redbuf[lane] : 0.f;
    float g = warp_reduce_sum(v);
    if (lane == 0) redbuf[0] = g;
  }
  __syncthreads();
  return redbuf[0];
}

// ====== Tìm max trong hàng (song song) ======
__device__ float row_max(const float* __restrict__ row, int V,
                         float* __restrict__ redbuf) {
  const int tid    = threadIdx.x;
  const int lane   = tid & (WARP_SIZE - 1);
  const int warp   = tid / WARP_SIZE;
  const int nwarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

  float lmax = -INFINITY;
  for (int i = tid; i < V; i += blockDim.x) {
    lmax = fmaxf(lmax, row[i]);
  }
  float wmax = warp_reduce_max(lmax);
  if (lane == 0) redbuf[warp] = wmax;
  __syncthreads();

  float mx = -INFINITY;
  if (warp == 0) {
    float v = (lane < nwarps) ? redbuf[lane] : -INFINITY;
    float g = warp_reduce_max(v);
    if (lane == 0) redbuf[0] = g;
  }
  __syncthreads();
  return redbuf[0];
}

// ====== Nucleus sampling không cần probindex_buf ======
// Mỗi block xử lý 1 hàng: probs[b, 0..V-1]
// coins[b] ∈ [0,1)
// 0 < topp < 1 (đã đảm bảo ở caller)
__global__ void k_sample_topp_rows_threshold(const float* __restrict__ probs,
                                             int V, int B,
                                             float topp,
                                             const float* __restrict__ coins,
                                             int* __restrict__ out_idx)
{
  const int b = blockIdx.x;
  if (b >= B) return;

  const float* row = probs + (size_t)b * V;
  const float coin = coins[b];

  // edge case nhỏ
  if (V <= 1) { if (threadIdx.x == 0) out_idx[b] = 0; return; }

  // Shared buffer cho reduce: đủ cho ≤32 warps
  __shared__ float redbuf[32];

  // --- (1) Tính max p để làm cận trên; và cận dưới theo cutoff CPU ---
  // cutoff = (1 - topp) / (V - 1) như bên CPU (giúp thu hẹp nhanh)
  float hi = row_max(row, V, redbuf);               // hi ≤ 1, chặt hơn 1.0
  float lo = (1.0f - topp) / (float)(V - 1);        // cận dưới lý thuyết

  // --- (2) Binary-search ngưỡng τ sao cho sum_{p_i >= τ} p_i >= topp ---
  // Invariant: f(lo) >= topp, f(hi) <= topp (điều chỉnh đôi chút)
  // Đảm bảo f(lo) >= topp:
  float sum_lo = row_sum_geq(row, V, lo, redbuf);
  if (sum_lo < topp) lo = 0.0f; // nới lỏng nếu cutoff quá cao
  // Đảm bảo f(hi) <= topp:
  float sum_hi = row_sum_geq(row, V, hi, redbuf);
  if (sum_hi > topp) {
    // hi vẫn quá thấp -> đẩy hi lên (max p), nhưng về lý thuyết hi=row_max => f(hi) = max_p >= topp?
    // Trường hợp topp rất nhỏ: nếu f(hi) > topp thì threshold sẽ nằm > hi;
    // ta cứ để binary-search trên [hi, 1.0].
    hi = fminf(1.0f, hi * 1.000001f);
  }

  // Binary search 16–20 vòng là dư
  float tau_lo = lo, tau_hi = fmaxf(hi, lo);
  for (int it = 0; it < 18; ++it) {
    float mid = 0.5f * (tau_lo + tau_hi);
    float s = row_sum_geq(row, V, mid, redbuf);
    if (s >= topp) tau_lo = mid; else tau_hi = mid;
  }
  float tau = tau_lo;

  // --- (3) Tổng khối lượng trong tập p_i >= tau ---
  float mass = row_sum_geq(row, V, tau, redbuf);
  if (!(mass > 0.f) || !isfinite(mass)) {
    // fallback an toàn: chọn argmax trong hàng
    if (threadIdx.x == 0) {
      // tìm argmax tuyến tính (song song cũng được, nhưng đơn giản)
      int best = 0; float mv = row[0];
      for (int i = 1; i < V; ++i) if (row[i] > mv) { mv = row[i]; best = i; }
      out_idx[b] = best;
    }
    return;
  }

  // --- (4) Bốc mẫu trong tập {i | p_i >= tau} theo r = coin * mass ---
  // Để đơn giản & chắc chắn, cho 1 thread quét tìm vị trí cắt.
  if (threadIdx.x == 0) {
    float r = coin * mass;
    float cdf = 0.f;
    int last = 0;
    for (int i = 0; i < V; ++i) {
      float p = row[i];
      if (p >= tau) {
        cdf += p;
        last = i;
        if (r < cdf) { out_idx[b] = i; return; }
      }
    }
    out_idx[b] = last; // dự phòng do sai số
  }
}

// ====== Launcher không cần probindex_buf ======
static inline void sample_topp_gpu_batch_no_buf(const float* probs_d, int B, int V,
                                                float topp, const float* d_coins,
                                                int* out_idx_d,
                                                hipStream_t stream = 0) {
  if (!(topp > 0.f && topp < 1.f) || B <= 0 || V <= 0) return;
  const int BLOCK = 256; // bội số của WARP_SIZE (64 trên AMD)
  dim3 grid(B), block(BLOCK);
  hipLaunchKernelGGL(k_sample_topp_rows_threshold, grid, block, 0, stream,
                     probs_d, V, B, topp, d_coins, out_idx_d);
}

void sample_gpu_batch(
    Sampler* sampler, float* logits_d, // [batch_size, vocab_size] in device memory 
    int B, int* out_idx_d, 
    float *coins, float* d_coins
) {
  const int V = sampler->vocab_size;

  if (sampler->temperature == 0.0f) {
    sample_argmax_gpu_batch(logits_d, out_idx_d, B, V);
  } else {
    scale_logits(logits_d, sampler->temperature, B, V);
    softmax_rows_gpu_batch_constlen(logits_d, B, V, V);

    get_coins_host(sampler, coins, B);
    HIP_CHECK(hipMemcpy(d_coins, coins, B * sizeof(float), hipMemcpyHostToDevice));

    if (sampler->topp <= 0 || sampler->topp >= 1) {
      sample_mult_gpu_batch(logits_d, B, V, d_coins, out_idx_d);
    } else {
      sample_topp_gpu_batch_no_buf(logits_d, B, V, sampler->topp, d_coins, out_idx_d);
    }
  }
}
