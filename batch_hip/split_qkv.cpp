// ====================================================================
// k_split_qkv_batch
//  - qkvB:   [B, total] với total = head_dim*(n_q + 2*n_kv)
//  - q_outB: [B, head_dim*n_q]
//  - k_all:  [L, MAX_BATCH, seq_len, kv_dim]  (kv_dim = head_dim*n_kv), truyền base ALL layer
//  - v_all:  như K
//  - positions: device ptr [B], vị trí pos[b] để ghi vào cache
//  - layer/MAX_BATCH_SIZE: để tính stride layer bên trong kernel
//
// Mỗi block.y = một sample b; block.x * threads xử lý theo khối 4-float (float4).
// Vector load (float4) nếu căn 16B; store scalar để an toàn.
// ====================================================================

__global__ void k_split_qkv_batch_f32q_bf16kv(const float*      __restrict__ qkvB,
                                              float*            __restrict__ q_outB,
                                              hip_bfloat16*     __restrict__ k_all,
                                              hip_bfloat16*     __restrict__ v_all,
                                              const int*        __restrict__ positions,
                                              int head_dim, int n_q, int n_kv,
                                              int B, int seq_len, int layer,
                                              int MAX_BATCH_SIZE, int sliding_window) {
  int b = blockIdx.y;
  if (b >= B) {
    return;
  }

  int kv_dim = head_dim * n_kv;
  int q_len  = head_dim * n_q;
  int total  = q_len + 2 * kv_dim;

  const size_t layer_stride_odd = MAX_BATCH_SIZE * seq_len * kv_dim;
  const size_t layer_stride_even = MAX_BATCH_SIZE * sliding_window * kv_dim;
  const size_t layer_stride = layer / 2 * layer_stride_odd + (layer + 1) / 2 * layer_stride_even;
  hip_bfloat16* k_layer = k_all + layer_stride;
  hip_bfloat16* v_layer = v_all + layer_stride;

  int pos = positions[b];
  if (sliding_window > 0 && layer % 2 == 0) {
    pos %= sliding_window;
  }
  size_t kv_row_off = (b * (layer % 2 ? seq_len : sliding_window) + pos) * kv_dim;

  const float*  src_b  = qkvB   + b * total;
  float*        qdst_b = q_outB + b * q_len;
  hip_bfloat16* kdst_b = k_layer + kv_row_off;
  hip_bfloat16* vdst_b = v_layer + kv_row_off;

  int threads  = blockDim.x;
  int idx4     = blockIdx.x * threads + threadIdx.x;
  int vec4_cnt = total >> 2;
  int tail     = total & 3;

  if (idx4 < vec4_cnt) {
    int j = idx4 << 2;
    Float4 v = gld_f32x4(src_b + j);

    #pragma unroll
    for (int t = 0; t < 4; ++t) {
      int   jj  = j + t;
      float val = (&v.x)[t];

      if (jj < q_len) {
        qdst_b[jj] = val;
      } else if (jj < q_len + kv_dim) {
        kdst_b[jj - q_len] = f32_to_bf16(val);
      } else {
        vdst_b[jj - q_len - kv_dim] = f32_to_bf16(val);
      }
    }
  }

  if (tail && threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) {
    for (int t = 0; t < tail; ++t) {
      int   j   = (vec4_cnt << 2) + t;
      float val = src_b[j];

      if (j < q_len) {
        qdst_b[j] = val;
      } else if (j < q_len + kv_dim) {
        kdst_b[j - q_len] = f32_to_bf16(val);
      } else {
        vdst_b[j - q_len - kv_dim] = f32_to_bf16(val);
      }
    }
  }
}

static inline void split_qkv_gpu_batch_devicepos_f32q_bf16kv(const float*  qkvB,
                                                             float*        q_outB,
                                                             hip_bfloat16* k_all,
                                                             hip_bfloat16* v_all,
                                                             const int*    d_positions,
                                                             int head_dim, int n_q, int n_kv,
                                                             int B, int seq_len, int layer,
                                                             int MAX_BATCH_SIZE, int sliding_window,
                                                             hipStream_t stream=0){
  int total = head_dim * (n_q + 2*n_kv);
  int vec4  = (total + 3) >> 2;
  int BS = 256;

  dim3 block(BS,1,1);
  dim3 grid(ceil_div(vec4,BS), B, 1);

  hipLaunchKernelGGL(k_split_qkv_batch_f32q_bf16kv, grid, block, 0, stream,
                     qkvB, q_outB, k_all, v_all, d_positions,
                     head_dim, n_q, n_kv, B, seq_len, layer, MAX_BATCH_SIZE, sliding_window);
}