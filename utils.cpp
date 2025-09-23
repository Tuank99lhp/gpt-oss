#define HIP_CHECK(cmd) do { \
  hipError_t e = (cmd);     \
  if (e != hipSuccess) {    \
    fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
            (int)e, hipGetErrorString(e), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

// ===== Helpers =====
#ifndef WARP_SIZE
#define WARP_SIZE warpSize        // 64 on AMD, 32 on NVIDIA
#endif
struct Float2 { float x, y; };
struct __align__(16) Float4 { float x,y,z,w; };

__device__ __forceinline__ int dmin(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int dceil_div(int n, int d) { return (n + d - 1) / d; }
__device__ __forceinline__ unsigned lane_id() { return threadIdx.x & (WARP_SIZE - 1); }
__device__ __forceinline__ unsigned warp_id() { return threadIdx.x / WARP_SIZE; }

// Warp-wide reduce sum (works for both 32/64 lanes)
__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) {
    v += __shfl_down(v, off, WARP_SIZE);
  }
  return v;
}

// warp reduce max
__inline__ __device__ float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down(v, offset));
    return v;
}

__global__ void k_set(float *x, float v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = v;
}

void set(float *x, float v, int n) {
  const int BS = 256, GS = (n + BS - 1) / BS;
  hipLaunchKernelGGL(k_set, dim3(GS), dim3(BS), 0, 0, x, v, n);
}

__global__ void k_set_vec(float *x, float *v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = v[i];
}

void set_vec(float *x, float *v, int n) {
  const int BS = 256, GS = (n + BS - 1) / BS;
  hipLaunchKernelGGL(k_set_vec, dim3(GS), dim3(BS), 0, 0, x, v, n);
}

int ceil_div(int n, int d) { return (n + d - 1) / d; }