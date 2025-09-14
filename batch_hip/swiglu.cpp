// ==== Kernel batched: dữ liệu dạng [B, row_stride] ====
// gate/up/out: [B, row_stride], liên tiếp theo chiều stride mỗi mẫu
__global__ void k_swiglu_gate_up_batched(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int B, int row_stride, float alpha, float clampv)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // index trong 1 mẫu
    const int b = blockIdx.y;                             // sample index
    if (b >= B || i >= row_stride) return;

    const size_t base = (size_t)b * (size_t)row_stride + (size_t)i;

    float g = gate[base];
    float u = up[base];

    // clamp (giữ nguyên đúng semantics bản đơn lẻ)
    g = fminf(g, clampv);
    u = fminf(fmaxf(u, -clampv), clampv);

    // SiLU(g) với hệ số alpha
    g = g * (1.0f / (1.0f + expf(-alpha * g)));

    // (u + 1) * SiLU(g)
    u = u + 1.0f;
    out[base] = g * u;
}

// ==== Wrapper: batched, stride = I (liền nhau theo mỗi mẫu) ====
static inline void swiglu_gpu_batch(
    float *gate, float *up, float *out,
    int I, float alpha, float clampv, int B)
{
    const int BS = 256;
    const int GX = (I + BS - 1) / BS;   // cột khối trong một mẫu
    dim3 grid(GX, B);
    dim3 block(BS);
    hipLaunchKernelGGL(k_swiglu_gate_up_batched, grid, block, 0, 0,
                       gate, up, out, B, I, alpha, clampv);
}