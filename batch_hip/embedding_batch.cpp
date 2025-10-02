struct fp32u { union { uint32_t u; float f; }; };

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t b) {
    fp32u v; v.u = (uint32_t)b << 16; return v.f;
}

// load 4 bf16 (8 bytes) từ địa chỉ p, chuyển sang float4
__device__ __forceinline__ float4 load_bf16x4_to_float4(const hip_bfloat16* p) {
    float4 r;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(p);
    if ((addr & 0x7) == 0) {
        // 8B aligned: 1x u64 load
        uint64_t pack = *reinterpret_cast<const uint64_t*>(p);
        uint32_t lo = static_cast<uint32_t>(pack);
        uint32_t hi = static_cast<uint32_t>(pack >> 32);

        uint16_t b0 = static_cast<uint16_t>(lo & 0xFFFFu);
        uint16_t b1 = static_cast<uint16_t>((lo >> 16) & 0xFFFFu);
        uint16_t b2 = static_cast<uint16_t>(hi & 0xFFFFu);
        uint16_t b3 = static_cast<uint16_t>((hi >> 16) & 0xFFFFu);

        r.x = bf16_bits_to_f32(b0);
        r.y = bf16_bits_to_f32(b1);
        r.z = bf16_bits_to_f32(b2);
        r.w = bf16_bits_to_f32(b3);
    } else {
        // unaligned: 4x 16-bit loads
        uint16_t b0 = *reinterpret_cast<const uint16_t*>(&p[0]);
        uint16_t b1 = *reinterpret_cast<const uint16_t*>(&p[1]);
        uint16_t b2 = *reinterpret_cast<const uint16_t*>(&p[2]);
        uint16_t b3 = *reinterpret_cast<const uint16_t*>(&p[3]);

        r.x = bf16_bits_to_f32(b0);
        r.y = bf16_bits_to_f32(b1);
        r.z = bf16_bits_to_f32(b2);
        r.w = bf16_bits_to_f32(b3);
    }
    return r;
}

//--- Kernel 1: phiên bản tổng quát (hỗn hợp vec4 + tail) ----------------------
__global__ void k_embedding_gather(const int* __restrict__ token_ids,
                                   const hip_bfloat16* __restrict__ emb, // [V, H] BF16
                                   float* __restrict__ out,               // [B, H] FP32
                                   int B, int H)
{
    int b = blockIdx.x;
    if (b >= B) return;

    int tid = threadIdx.x;
    int tok = token_ids[b];

    const hip_bfloat16* __restrict__ src = emb + 1ll * tok * H;
    float*       __restrict__ dst = out + 1ll * b   * H;

    // pha 1: xử lý theo nhóm 4 phần tử (8B load, 16B store)
    int H4 = H >> 2;
    float4* __restrict__ dst4 = reinterpret_cast<float4*>(dst);

    for (int i4 = tid; i4 < H4; i4 += blockDim.x) {
        const hip_bfloat16* p = src + (i4 << 2);
        float4 v = load_bf16x4_to_float4(p);
        dst4[i4] = v;
    }

    // pha 2: tail lẻ (<4) nếu có (ít xảy ra khi H%4==0, nhưng vẫn an toàn)
    int start_tail = H4 << 2;
    for (int i = start_tail + tid; i < H; i += blockDim.x) {
        uint16_t bx = *reinterpret_cast<const uint16_t*>(&src[i]);
        dst[i] = bf16_bits_to_f32(bx);
    }
}

inline void embedding_gather(float* out,
                             const hip_bfloat16* emb,
                             const int* d_tok,
                             int B, int H,
                             hipStream_t st)
{
    int BS = 256;
    hipLaunchKernelGGL(k_embedding_gather,
                       dim3(B), dim3(BS), 0, st,
                       d_tok, emb, out, B, H);
}

//--- Kernel 2: phiên bản vector hóa theo float4, giả định H % 4 == 0 ----------
__global__ void k_embedding_gather_vec4(const int* __restrict__ token_ids,
                                        const hip_bfloat16* __restrict__ emb, // [V, H] BF16
                                        float* __restrict__ out,              // [B, H] FP32
                                        int B, int H)
{
    int b = blockIdx.x;
    if (b >= B) return;

    int tid = threadIdx.x;

    int tok = token_ids[b];
    const hip_bfloat16* __restrict__ src = emb + 1ll * tok * H;
    float*       __restrict__ dst = out + 1ll * b   * H;

    int H4 = H >> 2;
    float4* __restrict__ dst4 = reinterpret_cast<float4*>(dst);

    for (int i4 = tid; i4 < H4; i4 += blockDim.x) {
        const hip_bfloat16* p = src + (i4 << 2);
        float4 v = load_bf16x4_to_float4(p);
        dst4[i4] = v;
    }
}

inline void embedding_gather_vec4(float* out,
                                  const hip_bfloat16* emb,
                                  const int* d_tok,
                                  int B, int H,
                                  hipStream_t st)
{
    int BS = 256; // bội số của 64 (wavefront) trên AMD
    hipLaunchKernelGGL(k_embedding_gather_vec4,
                       dim3(B), dim3(BS), 0, st,
                       d_tok, emb, out, B, H);
}