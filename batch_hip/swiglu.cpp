__global__ void k_swiglu_fused_gate_up_batched_bf16(
    const float*      __restrict__ mlp1_out, // [B*I*2], xen kẽ: g,u,g,u,...
    hip_bfloat16*     __restrict__ out,      // [B*I] (bf16)
    int B, int I,                             // batch size, intermediate_dim per sample
    float alpha, float clampv)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // 0..I-1 (theo 1 mẫu)
    const int b = blockIdx.y;                             // 0..B-1
    if (b >= B || i >= I) return;

    // chỉ số phẳng cho phần tử (b,i)
    const size_t f = (size_t)b * (size_t)I + (size_t)i;

    // đọc gate/up từ đầu vào xen kẽ (mỗi f tương ứng 2 phần tử liên tiếp)
    float g = mlp1_out[2 * f + 0];
    float u = mlp1_out[2 * f + 1];

    // clamp giống nguyên bản
    g = fminf(g, clampv);
    u = fminf(fmaxf(u, -clampv), clampv);

    // SiLU(g; alpha) = g * sigmoid(alpha*g)
    // có thể thay expf bằng __expf nếu bạn build với fast-math
    const float sig = 1.0f / (1.0f + expf(-alpha * g));
    g = g * sig;

    // (u + 1) * SiLU(g)
    u = u + 1.0f;
    out[f] = f32_to_bf16(g * u);
}

static inline void swiglu_fused_gpu_batch_bf16(
    const float*      mlp1_out,  // [B*I*2] xen kẽ g,u
    hip_bfloat16*     out,       // [B*I] bf16
    int I, int B,
    float alpha, float clampv,
    hipStream_t stream = 0)
{
    const int BS = 256;
    const int GX = (I + BS - 1) / BS;
    dim3 grid(GX, B);
    dim3 block(BS);

    hipLaunchKernelGGL(
        k_swiglu_fused_gate_up_batched_bf16,
        grid, block, 0, stream,
        mlp1_out, out, B, I, alpha, clampv
    );
}