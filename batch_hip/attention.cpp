// ===================== Batched Attention Kernels =====================

// ---------- k_attn_scores_batch ----------
// qB:        [B, n_heads*head_dim]
// k_layer:   [B, seq_len, kv_dim]   (đã offset đến layer l)  kv_dim = head_dim*n_kv
// mask:      [seq_len, seq_len]     hoặc nullptr
// attB:      [B, n_heads, row_stride]  (row_stride = seq_len+1)
// positions: [B] (device)
// Tính att[b,h,t] cho t = 0..pos[b] (CHƯA có sink)
__global__ void k_attn_scores_batch(const float* __restrict__ qB,
                                    const float* __restrict__ k_layer,
                                    const float* __restrict__ mask,
                                    float*       __restrict__ attB,
                                    const int*   __restrict__ positions,
                                    int head_dim, int kv_mul,
                                    int seq_len, int kv_dim,
                                    int n_heads, int row_stride,
                                    int B, int sliding_window)
{
  const int h = blockIdx.x;           // 0..n_heads-1
  const int b = blockIdx.y;           // 0..B-1
  if (h >= n_heads || b >= B) return;

  const int pos = positions[b];

  const float inv_sqrt_d = rsqrtf((float)head_dim);
  const float* qh = qB + (size_t)b * (size_t)n_heads * (size_t)head_dim
                      + (size_t)h * (size_t)head_dim;

  float* att_row = attB + ((size_t)b * (size_t)n_heads + (size_t)h) * (size_t)row_stride;

  // Thread block quét theo t với stride blockDim.x
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    const float* kt = k_layer
        + ((size_t)b * (size_t)seq_len + (size_t)t) * (size_t)kv_dim
        + (size_t)(h / kv_mul) * (size_t)head_dim;

    // dot(qh, kt)
    float s = 0.f;
    #pragma unroll 4
    for (int i = 0; i < head_dim; ++i) s = fmaf(qh[i], kt[i], s);
    s *= inv_sqrt_d;

    if (sliding_window > 0 && mask) {
      // mask row indexed by current pos (like code cũ)
      s += mask[(size_t)pos * (size_t)seq_len + (size_t)t];
    }
    att_row[t] = s;
  }
}

// ---------- k_attn_weighted_sum_batch ----------
// attB:   [B, n_heads, row_stride]  (đã softmax, gồm cả sink ở cột pos+1)
// v_layer:[B, seq_len, kv_dim]      (đã offset layer l)
// tbB:    [B, n_heads*head_dim]
__global__ void k_attn_weighted_sum_batch(const float* __restrict__ attB,
                                          const float* __restrict__ v_layer,
                                          float*       __restrict__ tbB,
                                          const int*   __restrict__ positions,
                                          int head_dim, int kv_mul,
                                          int row_stride, int seq_len,
                                          int kv_dim, int n_heads, int B)
{
  const int h = blockIdx.x;   // head
  const int b = blockIdx.y;   // batch
  if (h >= n_heads || b >= B) return;

  const int pos = positions[b];
  const int row_len = pos + 2; // gồm sink

  const float* att_row = attB + ((size_t)b * (size_t)n_heads + (size_t)h) * (size_t)row_stride;
  float* out_h = tbB + (size_t)b * (size_t)n_heads * (size_t)head_dim + (size_t)h * (size_t)head_dim;

  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float acc = 0.f;
    #pragma unroll 1
    for (int t = 0; t < row_len; ++t) {
      const float a = att_row[t];
      const float* vt = v_layer
          + ((size_t)b * (size_t)seq_len + (size_t)t) * (size_t)kv_dim
          + (size_t)(h / kv_mul) * (size_t)head_dim;
      acc = fmaf(a, vt[i], acc);
    }
    out_h[i] = acc;
  }
}

// ===================== Launchers =====================

// scores (batch)
static inline void attn_scores_gpu_batch(const float* qB,
                                         const float* k_layer,
                                         const float* mask,
                                         float* attB,
                                         const int* d_positions,
                                         int head_dim, int kv_mul,
                                         int seq_len, int kv_dim,
                                         int n_heads, int B,
                                         int sliding_window,
                                         hipStream_t stream = 0)
{
  const int BS = 256; // threads per block over time steps
  dim3 block(BS);
  dim3 grid(n_heads, B, 1);
  hipLaunchKernelGGL(k_attn_scores_batch, grid, block, 0, stream,
                     qB, k_layer, mask, attB, d_positions,
                     head_dim, kv_mul, seq_len, kv_dim,
                     n_heads, /*row_stride=*/seq_len+1, B, sliding_window);
}

// weighted sum (batch)
static inline void attn_weighted_sum_gpu_batch(const float* attB,
                                               const float* v_layer,
                                               float* tbB,
                                               const int* d_positions,
                                               int head_dim, int kv_mul,
                                               int seq_len, int kv_dim,
                                               int n_heads, int B,
                                               hipStream_t stream = 0)
{
  const int BS = 256; // threads over i
  dim3 block(BS);
  dim3 grid(n_heads, B, 1);
  hipLaunchKernelGGL(k_attn_weighted_sum_batch, grid, block, 0, stream,
                     attB, v_layer, tbB, d_positions, head_dim, kv_mul,
                     /*row_stride=*/seq_len+1, seq_len, kv_dim, n_heads, B);
}
