__global__ void k_embedding_gather(const int* __restrict__ token_ids,
                                   const float* __restrict__ emb, // [V, H]
                                   float* __restrict__ out,       // [B, H]
                                   int B, int H) {
  int b = blockIdx.x;
  if (b >= B) return;
  int tid = threadIdx.x;
  int tok = token_ids[b];
  const float* src = emb + 1ll * tok * H;
  float*       dst = out + 1ll * b   * H;

  // vectorize khi có thể
  int i = tid;
  for (; i + 3 < H; i += blockDim.x) {
    float4 v = *reinterpret_cast<const float4*>(&src[i]);
    *reinterpret_cast<float4*>(&dst[i]) = v;
  }
  for (; i < H; i += blockDim.x) dst[i] = src[i];
}

void embedding_gather(float* out, const float* emb, const int* d_tok, int B, int H, hipStream_t st) {
  int BS = 256;
  k_embedding_gather<<<B, BS, 0, st>>>(d_tok, emb, out, B, H);
}

// Giả định: hidden_dim % 4 == 0
// Layout: emb [V, H], out [B, H]
__global__ void k_embedding_gather_vec4(const int* __restrict__ token_ids,
                                        const float* __restrict__ emb,
                                        float* __restrict__ out,
                                        int B, int H)
{
    int b = blockIdx.x;
    if (b >= B) return;

    int tid = threadIdx.x;

    // Con trỏ đầu hàng nguồn/đích
    int tok = token_ids[b];
    const float* src = emb + 1ll * tok * H;
    float*       dst = out + 1ll * b   * H;

    // Vector view (128-bit)
    const float4* __restrict__ src4 = reinterpret_cast<const float4*>(src);
    float4* __restrict__       dst4 = reinterpret_cast<float4*>(dst);

    int H4 = H >> 2; // = H/4 phần tử float4

    // Mỗi thread xử lý các i4 = tid, tid+blockDim.x, ...
    for (int i4 = tid; i4 < H4; i4 += blockDim.x) {
        float4 v = src4[i4];  // 16B load (coalesced nếu các thread liên tiếp)
        dst4[i4] = v;         // 16B store
    }
}

// Launcher: 1 block cho mỗi sample, 256 threads (bội số của 64 trên AMD)
void embedding_gather_vec4(float* out, const float* emb, const int* d_tok,
                           int B, int H, hipStream_t st)
{
    int BS = 256; // 256 hoặc 128; nhớ bội số của 64 (wavefront)
    k_embedding_gather_vec4<<<B, BS, 0, st>>>(d_tok, emb, out, B, H);
}