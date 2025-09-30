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
__global__ void k_split_qkv_batch(const float* __restrict__ qkvB,
                                  float*       __restrict__ q_outB,
                                  float*       __restrict__ k_all,
                                  float*       __restrict__ v_all,
                                  const int*   __restrict__ positions,
                                  int head_dim, int n_q, int n_kv,
                                  int B, int seq_len, int layer,
                                  int MAX_BATCH_SIZE)
{
  const int b = blockIdx.y;                    // sample index
  if (b >= B) return;

  const int kv_dim  = head_dim * n_kv;
  const int q_len   = head_dim * n_q;
  const int kv_len  = kv_dim;                  // cho 1 (K) hoặc 1 (V)
  const int total   = q_len + 2 * kv_dim;

  // base offsets (theo layer) của cache K/V:
  const size_t layer_stride = (size_t)MAX_BATCH_SIZE * (size_t)seq_len * (size_t)kv_dim;
  float* k_layer = k_all + (size_t)layer * layer_stride;
  float* v_layer = v_all + (size_t)layer * layer_stride;

  const int pos = positions[b];
  const size_t kv_row_off = ((size_t)b * (size_t)seq_len + (size_t)pos) * (size_t)kv_dim;

  // base pointer cho sample b
  const float* __restrict__ src_b   = qkvB   + (size_t)b * (size_t)total;
  float*       __restrict__ qdst_b  = q_outB + (size_t)b * (size_t)q_len;
  float*       __restrict__ kdst_b  = k_layer + kv_row_off;
  float*       __restrict__ vdst_b  = v_layer + kv_row_off;

  // mapping theo block.x: xử lý theo đơn vị 4-float (float4)
  const int threads = blockDim.x;
  const int idx4    = blockIdx.x * threads + threadIdx.x;    // index theo float4
  const int vec4_cnt = total >> 2;                           // total/4
  const int tail     = total & 3;

  // ---- vectorized 4-float part ----
  if (idx4 < vec4_cnt) {
    const int j = idx4 << 2; // vị trí phần tử float trong [0..total)
    Float4 v = {0,0,0,0};

    // chỉ vector load nếu căn 16B
    const bool qkv_base16 = (((uintptr_t)src_b & 0xF) == 0);
    if (qkv_base16) {
      const Float4* __restrict__ p4 = reinterpret_cast<const Float4*>(src_b + j);
      v = *p4; // global float4 load
    } else {
      // fallback: scalar loads (hiếm)
      v.x = src_b[j + 0];
      v.y = src_b[j + 1];
      v.z = src_b[j + 2];
      v.w = src_b[j + 3];
    }

    // đẩy từng phần tử vào đích tương ứng
    // Q range: [0, q_len)
    // K range: [q_len, q_len+kv_len)
    // V range: [q_len+kv_len, total)
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int jj = j + t;
      const float val = (&v.x)[t];
      if (jj < q_len) {
        qdst_b[jj] = val;
      } else if (jj < q_len + kv_len) {
        kdst_b[jj - q_len] = val;
      } else {
        vdst_b[jj - q_len - kv_len] = val;
      }
    }
  }

  // ---- tail (≤3 phần tử cuối) cho thread 0 của mỗi block.x ----
  if (tail && threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) {
    for (int t = 0; t < tail; ++t) {
      const int j = (vec4_cnt << 2) + t;
      const float val = src_b[j];
      if (j < q_len) {
        qdst_b[j] = val;
      } else if (j < q_len + kv_len) {
        kdst_b[j - q_len] = val;
      } else {
        vdst_b[j - q_len - kv_len] = val;
      }
    }
  }
}

// --------------------- Launchers ---------------------

// positions đã ở device
static inline void split_qkv_gpu_batch_devicepos(const float* qkvB,
                                                 float* q_outB,
                                                 float* k_all,
                                                 float* v_all,
                                                 const int* d_positions, // device
                                                 int head_dim, int n_q, int n_kv,
                                                 int B, int seq_len,
                                                 int layer, int MAX_BATCH_SIZE,
                                                 hipStream_t stream=0)
{
  const int total = head_dim * (n_q + 2*n_kv);
  const int vec4  = (total + 3) >> 2;
  const int BS    = 256;
  dim3 block(BS, 1, 1);
  dim3 grid((vec4 + BS - 1) / BS, B, 1);
  hipLaunchKernelGGL(k_split_qkv_batch, grid, block, 0, stream,
                     qkvB, q_outB, k_all, v_all, d_positions,
                     head_dim, n_q, n_kv, B, seq_len, layer, MAX_BATCH_SIZE);
}

__global__ void k_split_qkv_batch_f32q_bf16kv(const float*      __restrict__ qkvB,
                                              float*            __restrict__ q_outB,
                                              hip_bfloat16*     __restrict__ k_all,
                                              hip_bfloat16*     __restrict__ v_all,
                                              const int*        __restrict__ positions,
                                              int head_dim,
                                              int n_q,
                                              int n_kv,
                                              int B,
                                              int seq_len,
                                              int layer,
                                              int MAX_BATCH_SIZE) {
  int b = blockIdx.y;
  if (b >= B) {
    return;
  }

  int kv_dim = head_dim * n_kv;
  int q_len  = head_dim * n_q;
  int total  = q_len + 2 * kv_dim;

  size_t layer_stride = MAX_BATCH_SIZE * seq_len * kv_dim;
  hip_bfloat16* k_layer = k_all + layer * layer_stride;
  hip_bfloat16* v_layer = v_all + layer * layer_stride;

  int pos = positions[b];
  size_t kv_row_off = (b * seq_len + pos) * kv_dim;

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
                                                             int head_dim,int n_q,int n_kv,
                                                             int B,int seq_len,int layer,int MAX_BATCH_SIZE,
                                                             hipStream_t stream=0){
  int total = head_dim * (n_q + 2*n_kv);
  int vec4  = (total + 3) >> 2;
  int BS = 256;

  dim3 block(BS,1,1);
  dim3 grid(ceil_div(vec4,BS), B, 1);

  hipLaunchKernelGGL(k_split_qkv_batch_f32q_bf16kv, grid, block, 0, stream,
                     qkvB, q_outB, k_all, v_all, d_positions,
                     head_dim, n_q, n_kv, B, seq_len, layer, MAX_BATCH_SIZE);
}