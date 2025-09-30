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

template<typename T>
__global__ void k_set(T *x, T v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = v;
}

template<typename T>
void set(T *x, T v, int n) {
  const int BS = 256, GS = (n + BS - 1) / BS;
  hipLaunchKernelGGL(k_set, dim3(GS), dim3(BS), 0, 0, x, v, n);
}

int ceil_div(int n, int d) { return (n + d - 1) / d; }

// MFMA accumulator type: vector of 4 float (matches builtin return type)
using v4f32 = float __attribute__((__vector_size__(4 * sizeof(float))));
using v4i16  = short __attribute__((__vector_size__(4 * sizeof(short))));

__device__ __forceinline__ float bf16_to_f32(hip_bfloat16 h) {
  return float(h);
}

__device__ __forceinline__ hip_bfloat16 f32_to_bf16(float x) {
  return hip_bfloat16(x);
}

__device__ __forceinline__ void gld_bf16x4(const hip_bfloat16* p,
                                           hip_bfloat16& a,
                                           hip_bfloat16& b,
                                           hip_bfloat16& c,
                                           hip_bfloat16& d) {
  const bool aligned8 = ((((uintptr_t)p) & 0x7) == 0);

  if (aligned8) {
    const uint64_t u  = *reinterpret_cast<const uint64_t*>(p);
    const uint32_t lo = static_cast<uint32_t>(u);
    const uint32_t hi = static_cast<uint32_t>(u >> 32);

    const uint16_t a16 = static_cast<uint16_t>(lo & 0xFFFFu);
    const uint16_t b16 = static_cast<uint16_t>(lo >> 16);
    const uint16_t c16 = static_cast<uint16_t>(hi & 0xFFFFu);
    const uint16_t d16 = static_cast<uint16_t>(hi >> 16);

    *reinterpret_cast<uint16_t*>(&a) = a16;
    *reinterpret_cast<uint16_t*>(&b) = b16;
    *reinterpret_cast<uint16_t*>(&c) = c16;
    *reinterpret_cast<uint16_t*>(&d) = d16;
  } else {
    a = p[0];
    b = p[1];
    c = p[2];
    d = p[3];
  }
}

__device__ __forceinline__ v4i16 pack_bf16x4_vec(hip_bfloat16 x0, hip_bfloat16 x1,
                                                 hip_bfloat16 x2, hip_bfloat16 x3) {
  // Lấy raw 16-bit của hip_bfloat16 rồi đóng gói vào short4
  uint16_t r0 = *reinterpret_cast<const uint16_t*>(&x0);
  uint16_t r1 = *reinterpret_cast<const uint16_t*>(&x1);
  uint16_t r2 = *reinterpret_cast<const uint16_t*>(&x2);
  uint16_t r3 = *reinterpret_cast<const uint16_t*>(&x3);
  v4i16 v = { (short)r0, (short)r1, (short)r2, (short)r3 };
  return v;
}

__device__ __forceinline__ Float4 gld_f32x4(const float* p) {
  Float4 v;
  if ((((uintptr_t)p) & 0xF) == 0) {
    v = *reinterpret_cast<const Float4*>(p);
  } else {
    v.x = p[0];
    v.y = p[1];
    v.z = p[2];
    v.w = p[3];
  }
  return v;
}