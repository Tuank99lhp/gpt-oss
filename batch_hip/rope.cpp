// ==================== Batched cos/sin + RoPE ====================
// - Tối ưu hóa thay cho vòng for per-batch hiện tại.
// - Tận dụng inv_freq precompute (1 lần), build cos/sin cho cả batch,
//   rồi áp RoPE cho Q và K-cache trong 2 kernel batched.
// - Không áp RoPE cho V (đúng như code của bạn).

// ======= (giữ nguyên thuật toán inv_freq của bạn) =======
__global__ void k_compute_concentration_and_inv_freq(
    float base, int head_dim, float scaling_factor, float initial_context_length,
    float ntk_beta, float ntk_alpha, float *concentration_out, float *inv_freq_out) {
  int d_half = head_dim / 2;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= d_half) return;

  float freq = powf(base, ((float)(2 * i)) / (float)head_dim);
  float concentration;
  if (scaling_factor > 1.0f) {
    concentration = 0.1f * logf(scaling_factor) + 1.0f;
    float low = d_half * logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) / logf(base);
    float high = d_half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(base);
    float interpolation = 1.0f / (scaling_factor * freq);
    float extrapolation = 1.0f / freq;
    float ramp = ((float)i - low) / (high - low);
    if (ramp < 0) ramp = 0;
    if (ramp > 1) ramp = 1;
    float mask = 1.0f - ramp;
    inv_freq_out[i] = interpolation * (1.0f - mask) + extrapolation * mask;
    if (i == 0) *concentration_out = concentration;
  } else {
    concentration = 1.0f;
    inv_freq_out[i] = 1.0f / freq;
    if (i == 0) *concentration_out = concentration;
  }
}

// ======= Build cos/sin cho cả batch (dựa trên positions[b]) =======
__global__ void k_build_cos_sin_batch(const int* __restrict__ positions,
                                      const float* __restrict__ inv_freq,
                                      float concentration,
                                      float* __restrict__ cosB, // [B, half]
                                      float* __restrict__ sinB, // [B, half]
                                      int B, int half) {
  int b = blockIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B || i >= half) return;
  float val = (float)positions[b] * inv_freq[i];
  float s, c; s = sinf(val); c = cosf(val);
  c *= concentration; s *= concentration;
  cosB[(size_t)b * half + i] = c;
  sinB[(size_t)b * half + i] = s;
}

// ======= Áp RoPE batched cho Q: qB [B, n_q*head_dim] =======
__global__ void k_rope_q_batch(float* __restrict__ qB,
                               const float* __restrict__ cosB,
                               const float* __restrict__ sinB,
                               int B, int n_q, int head_dim) {
  int b = blockIdx.y;        // sample
  int h = blockIdx.x;        // head idx (0..n_q-1)
  int tid = threadIdx.x;
  const int half = head_dim >> 1;
  if (b >= B || h >= n_q) return;

  const size_t base = ((size_t)b * n_q + h) * (size_t)head_dim;

  for (int i = tid; i < half; i += blockDim.x) {
    float c = cosB[(size_t)b * half + i];
    float s = sinB[(size_t)b * half + i];
    float x1 = qB[base + i];
    float x2 = qB[base + half + i];
    float o1 = x1 * c - x2 * s;
    float o2 = x2 * c + x1 * s;
    qB[base + i]          = o1;
    qB[base + half + i]   = o2;
  }
}

static inline void rope_ensure_invfreq(int head_dim, float base,
                                       float scaling_factor, float initial_context_length,
                                       float ntk_beta, float ntk_alpha,
                                       hipStream_t stream = 0)
{
  int device_id = 0;
  HIP_CHECK(hipGetDevice(&device_id));
  BatchState *g_batch_state = &batch_states[device_id];

  float *g_inv_freq_dev = g_batch_state->g_inv_freq_dev;
  float *d_conc = g_batch_state->d_conc;

  int    &g_cached_hd    = g_batch_state->g_cached_hd;
  float  &g_cached_base  = g_batch_state->g_cached_base;
  float  &g_cached_scale = g_batch_state->g_cached_scale;
  float  &g_cached_ic    = g_batch_state->g_cached_ic;
  float  &g_cached_b     = g_batch_state->g_cached_b;
  float  &g_cached_a     = g_batch_state->g_cached_a;
  float  &g_concentration = g_batch_state->g_concentration;

  const bool need_rebuild =
      (g_inv_freq_dev == nullptr) ||
      (g_cached_hd   != head_dim) ||
      (g_cached_base != base) ||
      (g_cached_scale!= scaling_factor) ||
      (g_cached_ic   != initial_context_length) ||
      (g_cached_b    != ntk_beta) ||
      (g_cached_a    != ntk_alpha);

  if (!need_rebuild) return;

  // if (g_inv_freq_dev) HIP_CHECK(hipFree(g_inv_freq_dev));
  const int half = head_dim >> 1;

  const int BS = 256;
  dim3 grid((half + BS - 1) / BS);
  dim3 block(BS);
  hipLaunchKernelGGL(k_compute_concentration_and_inv_freq, grid, block, 0, stream,
                     base, head_dim, scaling_factor, initial_context_length,
                     ntk_beta, ntk_alpha, d_conc, g_inv_freq_dev);
  HIP_CHECK(hipMemcpyAsync(&g_concentration, d_conc, sizeof(float), hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  // HIP_CHECK(hipFree(d_conc));

  g_cached_hd    = head_dim;
  g_cached_base  = base;
  g_cached_scale = scaling_factor;
  g_cached_ic    = initial_context_length;
  g_cached_b     = ntk_beta;
  g_cached_a     = ntk_alpha;
}

// ======= API tiện dụng: build cos/sin & áp RoPE (batched) =======
static inline void rope_build_cos_sin_batch(float* cosB, float* sinB,
                                            const int* d_positions,
                                            int head_dim, int B,
                                            hipStream_t stream = 0)
{
  int device_id = 0;
  HIP_CHECK(hipGetDevice(&device_id));
  BatchState *g_batch_state = &batch_states[device_id];
  
  float *g_inv_freq_dev = g_batch_state->g_inv_freq_dev;
  float g_concentration = g_batch_state->g_concentration;
  
  const int half = head_dim >> 1;
  const int BS = 256;
  dim3 block(BS);
  dim3 grid((half + BS - 1) / BS, B, 1);
  hipLaunchKernelGGL(k_build_cos_sin_batch, grid, block, 0, stream,
                     d_positions, g_inv_freq_dev, g_concentration, cosB, sinB, B, half);
}

static inline void rope_apply_q_batch(float* qB, const float* cosB, const float* sinB,
                                      int B, int n_q, int head_dim,
                                      hipStream_t stream = 0)
{
  const int BS = 256; // tile theo i (half)
  dim3 block(BS);
  dim3 grid(n_q, B, 1);
  hipLaunchKernelGGL(k_rope_q_batch, grid, block, 0, stream,
                     qB, cosB, sinB, B, n_q, head_dim);
}

// k_all: [L, MAX_BATCH, seq_len, kv_dim] (bf16), cosB/sinB: [B, head_dim/2] (f32)
__global__ void k_rope_k_batch_bf16(hip_bfloat16*   __restrict__ k_all,
                                    const float*    __restrict__ cosB,
                                    const float*    __restrict__ sinB,
                                    const int*      __restrict__ positions,
                                    int B, int n_kv, int head_dim,
                                    int seq_len, int kv_dim,
                                    int layer, int MAX_BATCH_SIZE, int sliding_window) {
  const int b = blockIdx.y;
  const int h = blockIdx.x;
  const int tid = threadIdx.x;

  if (b >= B || h >= n_kv) {
    return;
  }

  const int half = head_dim >> 1;

  const size_t layer_stride_odd = MAX_BATCH_SIZE * seq_len * kv_dim;
  const size_t layer_stride_even = MAX_BATCH_SIZE * sliding_window * kv_dim;
  const size_t layer_stride = layer / 2 * layer_stride_odd + (layer + 1) / 2 * layer_stride_even;

  hip_bfloat16* k_layer = k_all + layer_stride;

  int pos = positions[b];
  if (sliding_window > 0 && layer % 2 == 0) {
    pos %= sliding_window;
  }

  const size_t row = (b * (layer % 2 ? seq_len : sliding_window) + pos) * kv_dim + h * head_dim;

  for (int i = tid; i < half; i += blockDim.x) {
    const float c = cosB[b * half + i];
    const float s = sinB[b * half + i];

    const float x1 = bf16_to_f32(k_layer[row + i]);
    const float x2 = bf16_to_f32(k_layer[row + half + i]);

    const float o1 = x1 * c - x2 * s;
    const float o2 = x2 * c + x1 * s;

    k_layer[row + i]         = f32_to_bf16(o1);
    k_layer[row + half + i]  = f32_to_bf16(o2);
  }
}

static inline void rope_apply_k_batch_bf16(hip_bfloat16* k_all,
                                           const float*  cosB,
                                           const float*  sinB,
                                           const int*    d_positions,
                                           int B,
                                           int n_kv,
                                           int head_dim,
                                           int seq_len,
                                           int kv_dim,
                                           int layer,
                                           int MAX_BATCH_SIZE,
                                           int sliding_window,
                                           hipStream_t stream = 0) {
  const int BS = 256;
  dim3 block(BS);
  dim3 grid(n_kv, B, 1);

  hipLaunchKernelGGL(
    k_rope_k_batch_bf16, grid, block, 0, stream,
    k_all, cosB, sinB, d_positions,
    B, n_kv, head_dim, seq_len, kv_dim, layer, 
    MAX_BATCH_SIZE, sliding_window
  );
}