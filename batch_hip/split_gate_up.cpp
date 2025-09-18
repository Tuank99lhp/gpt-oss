__global__ void k_split_gate_up(const float *mlp1_out, float *gate, float *up, int intermediate_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < intermediate_dim) {
        gate[i] = mlp1_out[i*2];
        up[i]   = mlp1_out[i*2 + 1];
    }
}

void split_gate_up(const float *mlp1_out, float *gate, float *up, int intermediate_dim) {
    const int BS = 256, GS = (intermediate_dim + BS - 1) / BS;
    hipLaunchKernelGGL(k_split_gate_up, dim3(GS), dim3(BS), 0, 0, mlp1_out, gate, up, intermediate_dim);
}