// split_qkv_batched.cu
// Batched split Q/K/V for HIP (no external libs).
// Layout per sample (input qkv):
//   [ Q (n_q*head_dim), K (n_kv*head_dim), V (n_kv*head_dim) ]
//
// Output:
//   - Q: q_batch[b, q_size]  (contiguous)
//   - K,V: scatter vào KV-cache tại (b_idx, pos) cho layer hiện tại:
//       k_layer_base[ (b_idx*seq_len + pos) * kv_size + i ] ← K_i
//       v_layer_base[ (b_idx*seq_len + pos) * kv_size + i ] ← V_i
//
// Notes:
//   - Dùng 3 kernel riêng cho Q / K / V để tối đa coalescing và tránh branch.
//   - Không giả định alignment; scalar copy an toàn cho mọi kích thước.
//   - Bạn phải truyền base pointer LAYER-LOCAL (đã cộng offset theo layer).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef HIP_CHECK
#define HIP_CHECK(cmd) do { \
  hipError_t _e = (cmd);    \
  if (_e != hipSuccess) {   \
    fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
            (int)_e, hipGetErrorString(_e), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)
#endif

// -------------------------- Kernels --------------------------


__global__ void k_split_qkv_batch_hostidx(const float *qkv,
                                          float *q_out,
                                          float *k_base, float *v_base,
                                          int head_dim, int n_q, int n_kv,
                                          int batch_size,
                                          int seq_len,
                                          const int *indx,        // map batch b -> b_idx (toàn cục)
                                          const int *positions,   // positions[b_idx]
                                          int max_batch_size,     // MAX_BATCH_SIZE
                                          int kv_dim)             // = head_dim * n_kv
{
  const int total = head_dim * (n_q + 2 * n_kv);
  const int q_len = head_dim * n_q;
  const int k_len = head_dim * n_kv;
  const int v_len = head_dim * n_kv;

  int i = blockIdx.x * blockDim.x + threadIdx.x; // phần tử trong qkv của 1 sample
  int b = blockIdx.y;                             // sample trong batch

  if (b >= batch_size || i >= total) return;

  // Pointers theo sample trong batch
  const float *qkv_b = qkv + (size_t)b * (size_t)total;
  float *qb = q_out + (size_t)b * (size_t)q_len;

  // Map sang chỉ số global + pos để tính offset cache
  int b_idx = indx[b];                 // 0..MAX_BATCH_SIZE-1 (slot toàn cục)
  int pos   = positions[b_idx];        // 0..seq_len-1

  // base offset trong cache (đÃ offset theo layer trước khi truyền vào kernel)
  size_t cache_row = (size_t)b_idx * (size_t)seq_len + (size_t)pos;
  size_t base_off  = cache_row * (size_t)kv_dim; // nơi ghi K/V cho sample này

  float val = qkv_b[i];
  if (i < q_len) {
    // Q ghi ra q_out (liên tục theo batch)
    qb[i] = val;
  } else if (i < q_len + k_len) {
    // K ghi thẳng vào KV-cache
    size_t j = (size_t)i - (size_t)q_len; // 0..k_len-1
    k_base[base_off + j] = val;
  } else {
    // V ghi thẳng vào KV-cache
    size_t j = (size_t)i - (size_t)q_len - (size_t)k_len; // 0..v_len-1
    v_base[base_off + j] = val;
  }
}

static inline void split_qkv_gpu_batch_hostidx(const float *qkv,
                                               float *q_out,
                                               float *k_cache_layer_base, // đã cộng offset layer
                                               float *v_cache_layer_base, // đã cộng offset layer
                                               int head_dim, int n_q, int n_kv,
                                               int batch_size,
                                               int seq_len,
                                               const int *indx,          // device pointer
                                               const int *positions,     // device pointer
                                               int max_batch_size,
                                               hipStream_t stream = 0)
{

  int *d_bidx = nullptr, *d_pos = nullptr;
  HIP_CHECK(hipMalloc(&d_bidx, batch_size * sizeof(int)));
  HIP_CHECK(hipMemcpyAsync(d_bidx, indx, batch_size * sizeof(int),
                           hipMemcpyHostToDevice, stream));

  HIP_CHECK(hipMalloc(&d_pos, max_batch_size * sizeof(int)));
  HIP_CHECK(hipMemcpyAsync(d_pos, positions, max_batch_size * sizeof(int),
                           hipMemcpyHostToDevice, stream));

  const int kv_dim = head_dim * n_kv;
  const int total  = head_dim * (n_q + 2 * n_kv);

  const int BS = 256;
  dim3 block(BS, 1);
  dim3 grid((total + BS - 1) / BS, batch_size);

  hipLaunchKernelGGL(k_split_qkv_batch_hostidx, grid, block, 0, 0,
                     qkv, q_out,
                     k_cache_layer_base, v_cache_layer_base,
                     head_dim, n_q, n_kv,
                     batch_size, seq_len,
                     d_bidx, d_pos, max_batch_size,
                     kv_dim);

  HIP_CHECK(hipFree(d_bidx));
  HIP_CHECK(hipFree(d_pos));
}