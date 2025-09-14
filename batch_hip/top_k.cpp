// ================= Batched Top-K for small K (1..8) =================

// ---- one block per row (b); each thread keeps local top-K in registers ----
__global__ void k_topk_batch_smallK(const float* __restrict__ scoresBN, // [B,N]
                                    float*       __restrict__ topk_vals, // [B,K]
                                    int*         __restrict__ topk_idx,  // [B,K]
                                    int N, int K, int B)
{
  const int b = blockIdx.x;
  if (b >= B) return;
  const float* row = scoresBN + (size_t)b * (size_t)N;

  // local topK in registers
  const int KMAX = 8;
  float v[KMAX]; int ix[KMAX];
  #pragma unroll
  for (int r=0;r<KMAX;++r){ v[r] = -INFINITY; ix[r] = -1; }

  // scan strided
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    float x = row[j];
    // insert x into local top-K (descending)
    int r = min(K, KMAX) - 1;
    if (x > v[r]) {
      #pragma unroll
      for (; r > 0 && x > v[r-1]; --r) { v[r] = v[r-1]; ix[r] = ix[r-1]; }
      v[r] = x; ix[r] = j;
    }
  }

  // write locals to shared
  extern __shared__ unsigned char smem[];
  float* svals = (float*)smem;
  int*   sidx  = (int*)(svals + (size_t)blockDim.x * (size_t)K);
  const int base = threadIdx.x * K;
  #pragma unroll
  for (int r=0;r<K;++r) {
    svals[base + r] = v[r];
    sidx [base + r] = ix[r];
  }
  __syncthreads();

  // select global top-K from (blockDim.x * K) candidates
  if (threadIdx.x == 0) {
    const int C = blockDim.x * K; // candidate count
    for (int out = 0; out < K; ++out) {
      int    best_i = -1;
      float  best_v = -INFINITY;
      for (int c = 0; c < C; ++c) {
        float vc = svals[c];
        if (vc > best_v) { best_v = vc; best_i = c; }
      }
      topk_vals[(size_t)b * (size_t)K + out] = best_v;
      topk_idx [(size_t)b * (size_t)K + out] = (best_i >= 0) ? sidx[best_i] : -1;
      if (best_i >= 0) svals[best_i] = -INFINITY; // remove chosen
    }
  }
}

// -------------------- Host helpers --------------------
static inline void topk_gpu_batch(float* d_topk_vals, int* d_topk_idx,
                                  const float* d_scoresBN,
                                  int B, int N, int K,
                                  hipStream_t stream=0)
{
  // guard
  if (K <= 0) return;
  const int KMAX = 8;
  if (K > KMAX) {
    // Fallback: chạy nhiều vòng argmax K lần (đơn giản, vẫn batched); để ngắn gọn ta assume K<=8.
    fprintf(stderr, "[topk_gpu_batch] K=%d > %d not supported in smallK kernel.\n", K, KMAX);
    abort();
  }

  const int BS = 256; // multiple of 64 (AMD wavefront)
  dim3 grid(B), block(BS);
  size_t shmem = (size_t)BS * (size_t)K * (sizeof(float) + sizeof(int));

  hipLaunchKernelGGL(k_topk_batch_smallK, grid, block, shmem, stream,
                     d_scoresBN, d_topk_vals, d_topk_idx, N, K, B);
}
