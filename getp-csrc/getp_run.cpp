#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <omp.h>

#include "getp_eval.cpp"
#include "../utils.cpp"

#ifndef GETP_RUN
#define GETP_RUN

typedef struct {
    // ==== Per-request state (host) ====
    int* positions;              // [batch_size]  vị trí hiện tại của từng sequence
    int* num_prompt_tokens;      // [batch_size]  độ dài prompt của từng sequence
    int** prompt_tokens;         // [batch_size][] mảng token prompt (malloc/free trên host)
    int* current_tokens;         // [batch_size]  token hiện tại của từng sequence
    bool* finished;              // [batch_size]  cờ kết thúc cho từng sequence
    int* req_ids;

    // Logits
    float*  batch_logits;        // [batch_size, vocab_size] logits trên device (contiguous)

    // ==== Pre-allocated buffers for batched operations (device) ====
    // Kích thước sử dụng: [max_batch_size, ...]; bạn cấp phát theo max_batch_size, dùng batch_size phần đầu.

    // Emb/Residual stream
    float* batch_x;              // [max_batch_size, hidden_dim]
    hip_bfloat16* batch_t;              // [max_batch_size, hidden_dim]

    // Attention projections
    float* batch_qkv;            // [max_batch_size, head_dim * (n_attn_heads + 2*n_kv_heads)]
    float* batch_q;              // [max_batch_size, head_dim * n_attn_heads]

    // Attention caches (cho toàn bộ lịch sử 0..pos, theo layer & batch)
    // Layout gợi ý: [n_layers, max_batch_size, seq_len, kv_dim]
    hip_bfloat16* batch_k;
    hip_bfloat16* batch_v;

    // Attention scores & outputs
    float* batch_att;            // [max_batch_size, n_attn_heads, (seq_len + 1)]  // +1 để append sink
    hip_bfloat16* batch_tb;      // [max_batch_size, head_dim * n_attn_heads]
    float* batch_tb2;            // [max_batch_size, hidden_dim]

    // ==== MLP / MoE buffers (device) ====
    float* batch_router_score;   // [max_batch_size, n_experts]
    float* batch_topk_v;         // [max_batch_size, experts_per_token]
    int*   batch_topk_i;         // [max_batch_size, experts_per_token]

    float* mask;

    int* d_current_tokens;
    int* d_positions;
    float* cosB;
    float* sinB;

    int* next_tokens;
    int* d_next_tokens;

    float* coins;
    float* d_coins;

    hip_bfloat16* d_w_token_embedding_table_bf16;
    hip_bfloat16* d_w_qkv_bf16;
    hip_bfloat16* d_w_o_bf16;
    hip_bfloat16* d_w_router_bf16;
    hip_bfloat16* d_w_mlp1_bf16;
    hip_bfloat16* d_w_mlp2_bf16;
    hip_bfloat16* d_w_out_bf16;
    
    // ======= Bộ nhớ cache inv_freq (tạo 1 lần, dùng lại) =======
    float* g_inv_freq_dev; // [half] on device
    float* d_conc;
    int    g_cached_hd    = 0;
    float  g_cached_base  = 0.f, g_cached_scale = 0.f, g_cached_ic = 0.f, g_cached_b = 0.f, g_cached_a = 0.f;
    float  g_concentration = 1.f;

    int*    h_counts;
    int*    d_counts;
    int*    d_idx_in_batch;
    hip_bfloat16** h_in_ptrs;
    hip_bfloat16** d_in_ptrs;
    float*  d_out;
    
    float* batch_mlp1_out;       // [max_batch_size, 2 * intermediate_dim]
    hip_bfloat16* batch_gate_up;        // [max_batch_size, intermediate_dim]

} BatchState;

int NUM_GPUS = 8;
int MAX_BATCH_SIZE = 890;
BatchState* batch_states = NULL;
TransformerWeights* transformer_weights = NULL;

bool is_120b_model(Config *cfg) {
  return cfg->n_layers == 36;
}

#include "../forward.cpp"
#include "../forward_120b.cpp"
#include "../sample.cpp"

// ------------------------ GPU allocations / deallocations ------------------------

template<typename T>
static void to_device(T **dptr, const T *hptr, long long nbytes) {
  HIP_CHECK(hipMalloc((void**)dptr, nbytes));
  HIP_CHECK(hipMemcpy(*dptr, hptr, nbytes, hipMemcpyHostToDevice));
}

template<typename T>
static void alloc_device(T **dptr, long long nbytes, T fill, bool do_set = false) {
  HIP_CHECK(hipMalloc((void**)dptr, nbytes));
  if (do_set) {
    long long n = nbytes / sizeof(T);
    set(*dptr, fill, n);
  }
}

template<typename T>
void free_device(T *&p) {
  if (p) {
    hipFree(p);
    p = nullptr; 
  } 
}

template <typename T>
void free_host(T *&p) {
  if (p) {
    free(p);
    p = nullptr; 
  }
}

// -------------------------- File mapping (host) --------------------------

static void convert_fp32_to_bf16_host(const float *src, hip_bfloat16 *dst, long long N) {
  #pragma omp parallel for
  for (long long i = 0; i < N; i++) {
      dst[i] = hip_bfloat16(src[i]);
      // if (i < 1000) printf("Converting fp32 to bf16: %.32f -> %.32f\n", src[i], (float)dst[i]);
  }
}

static void convert_fp32_to_bf16_host_and_transpose(const float *src, hip_bfloat16 *dst, int n, int rows, int cols) {
  // src: [n, rows, cols] -> dst: [n, cols, rows]
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    const float *s = src + (long long)i * rows * cols;
    hip_bfloat16 *d = dst + (long long)i * rows * cols;
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        d[c * rows + r] = hip_bfloat16(s[r * cols + c]);
      }
    }
  }
}

long long max3(long long a, long long b, long long c) {
  long long v = a;
  if (b > v) v = b;
  if (c > v) v = c;
  return v;
}

void memory_map_weights_gpu(TransformerWeights *w, Config *cfg, float *ptr, BatchState *g_batch_state) {
  int head_dim = cfg->head_dim;
  int n_layers = cfg->n_layers;
  int n_experts = cfg->n_experts;

  long long size_tmp = max3(
    1ll * cfg->vocab_size * cfg->hidden_dim,
    1ll * n_layers * cfg->hidden_dim * (head_dim * (cfg->n_attn_heads + 2 * cfg->n_kv_heads)),
    1ll * n_layers * n_experts * 2 * cfg->intermediate_dim * cfg->hidden_dim
  );
  hip_bfloat16 *tmp = (hip_bfloat16*)malloc(size_tmp * sizeof(hip_bfloat16));
  if (!tmp) {
      fprintf(stderr, "OOM: cannot allocate host FP16 buffer for tmp\n");
      exit(1);
  }

  convert_fp32_to_bf16_host(ptr, tmp, 1ll * cfg->vocab_size * cfg->hidden_dim);
  to_device(&g_batch_state->d_w_token_embedding_table_bf16, tmp, 1ll * cfg->vocab_size * cfg->hidden_dim * sizeof(hip_bfloat16));
  ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;

  convert_fp32_to_bf16_host_and_transpose(ptr, tmp, 1, cfg->vocab_size, cfg->hidden_dim);
  to_device(&g_batch_state->d_w_out_bf16, tmp, 1ll * cfg->vocab_size * cfg->hidden_dim * sizeof(hip_bfloat16));
  ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;

  to_device(&w->rms_attn_w, ptr, 1ll * n_layers * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim;
  to_device(&w->rms_ffn_w, ptr, 1ll * n_layers * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim;
  to_device(&w->rms_out_w, ptr, 1ll * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * cfg->hidden_dim;

  convert_fp32_to_bf16_host_and_transpose(
    ptr, tmp, n_layers,
    head_dim * (cfg->n_attn_heads + 2 * cfg->n_kv_heads), 
    cfg->hidden_dim
  );
  to_device(
    &g_batch_state->d_w_qkv_bf16, tmp, 
    1ll * n_layers * cfg->hidden_dim * head_dim *
    (cfg->n_attn_heads + 2 * cfg->n_kv_heads) * sizeof(hip_bfloat16)
  );
  ptr += 1ll * n_layers * cfg->hidden_dim *
         (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);

  to_device(&w->b_qkv, ptr,
            1ll * n_layers * (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads) *
            sizeof(float));
  ptr += 1ll * n_layers *
         (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);

  convert_fp32_to_bf16_host_and_transpose(
    ptr, tmp, n_layers,
    cfg->hidden_dim,
    head_dim * cfg->n_attn_heads
  );
  to_device(
    &g_batch_state->d_w_o_bf16, tmp,
    1ll * n_layers * (head_dim * cfg->n_attn_heads) * cfg->hidden_dim * sizeof(hip_bfloat16)
  );
  ptr += 1ll * n_layers * (head_dim * cfg->n_attn_heads) * cfg->hidden_dim;

  to_device(&w->b_o, ptr, 1ll * n_layers * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim;
  to_device(&w->attn_sinks, ptr, 1ll * n_layers * cfg->n_attn_heads * sizeof(float));
  ptr += 1ll * n_layers * cfg->n_attn_heads;

  convert_fp32_to_bf16_host_and_transpose(ptr, tmp, n_layers, n_experts, cfg->hidden_dim);
  to_device(&g_batch_state->d_w_router_bf16, tmp, 1ll * n_layers * n_experts * cfg->hidden_dim * sizeof(hip_bfloat16));
  ptr += 1ll * n_layers * cfg->hidden_dim * n_experts;
  
  to_device(&w->b_router, ptr, 1ll * n_layers * n_experts * sizeof(float));
  ptr += 1ll * n_layers * n_experts;
    
  long long elems_w_mlp1 = 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim * cfg->hidden_dim;

  // convert_fp32_to_bf16_host(ptr, tmp, elems_w_mlp1);
  convert_fp32_to_bf16_host_and_transpose(ptr, tmp, n_layers * n_experts, 2 * cfg->intermediate_dim, cfg->hidden_dim);

  to_device(&g_batch_state->d_w_mlp1_bf16, tmp, elems_w_mlp1 * sizeof(hip_bfloat16));
  ptr += elems_w_mlp1;
  
  to_device(&w->b_mlp1, ptr, 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim * sizeof(float));
  ptr += 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim;

  long long elems_w_mlp2 = 1ll * n_layers * n_experts * cfg->hidden_dim * cfg->intermediate_dim;

  // convert_fp32_to_bf16_host(ptr, tmp, elems_w_mlp2);
  convert_fp32_to_bf16_host_and_transpose(ptr, tmp, n_layers * n_experts, cfg->hidden_dim, cfg->intermediate_dim);

  to_device(&g_batch_state->d_w_mlp2_bf16, tmp, elems_w_mlp2 * sizeof(hip_bfloat16));
  ptr += elems_w_mlp2;
  
  free(tmp);
  
  to_device(&w->b_mlp2, ptr, 1ll * n_layers * n_experts * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * n_experts * cfg->hidden_dim;
}

static void free_weights_gpu(TransformerWeights &w, BatchState *g_batch_state) {
  free_device(g_batch_state->d_w_token_embedding_table_bf16); 
  
  free_device(w.rms_attn_w); 
  free_device(w.rms_ffn_w); 
    
  // free_device(w.w_qkv); 
  free_device(g_batch_state->d_w_qkv_bf16);
  free_device(w.b_qkv);

  // free_device(w.w_o); 
  free_device(g_batch_state->d_w_o_bf16);
  free_device(w.b_o); 
    
  free_device(w.attn_sinks); 
    
  // free_device(w.w_router); 
  free_device(g_batch_state->d_w_router_bf16);
  free_device(w.b_router);
    
  free_device(g_batch_state->d_w_mlp1_bf16);
  free_device(w.b_mlp1); 

  free_device(g_batch_state->d_w_mlp2_bf16);
  free_device(w.b_mlp2); 
    
  free_device(w.rms_out_w); 
  // free_device(w.out);
  free_device(g_batch_state->d_w_out_bf16);
}

// -------------------------- BatchState management --------------------------

static void alloc_batchstate_on_device(BatchState &bs, const Config &c) {
  // ====== Host-side per-request arrays ======
  bs.positions         = (int  *)calloc(MAX_BATCH_SIZE, sizeof(int));
  bs.num_prompt_tokens = (int  *)calloc(MAX_BATCH_SIZE, sizeof(int));
  bs.prompt_tokens     = (int **)calloc(MAX_BATCH_SIZE, sizeof(int*)); // các phần tử sẽ malloc/free trong inference
  bs.current_tokens    = (int  *)calloc(MAX_BATCH_SIZE, sizeof(int));
  bs.finished          = (bool *)calloc(MAX_BATCH_SIZE, sizeof(bool));
  bs.req_ids           = (int  *)calloc(MAX_BATCH_SIZE, sizeof(int));

  // ====== Device-side batched buffers ======
  const int H   = c.hidden_dim;
  const int D   = c.head_dim;
  const int Hq  = c.n_attn_heads;
  const int Hkv = c.n_kv_heads;
  const int kv_dim = D * Hkv;
  const int qkv_tot = D * (Hq + 2 * Hkv);
  const int B = MAX_BATCH_SIZE;

  // Embedding/residual stream
  alloc_device(&bs.batch_x, 1ll * B * H * sizeof(float), 0.f, true);
  alloc_device(&bs.batch_t, 1ll * B * H * sizeof(hip_bfloat16), hip_bfloat16(0.f), true);

  // Attention projections
  alloc_device(&bs.batch_qkv, 1ll * B * qkv_tot * sizeof(float), 0.f, true);
  alloc_device(&bs.batch_q, 1ll * B * (D * Hq) * sizeof(float), 0.f, true);

  // Attention caches: [n_layers, MAX_BATCH_SIZE, seq_len, kv_dim]
  {
    const long long cache_elems = 1ll * c.n_layers / 2 * B * (c.seq_len + c.sliding_window) * kv_dim;
    alloc_device(&bs.batch_k, cache_elems * sizeof(hip_bfloat16), hip_bfloat16(0.f), true);
    alloc_device(&bs.batch_v, cache_elems * sizeof(hip_bfloat16), hip_bfloat16(0.f), true);
  }

  // Attention scores & outputs
  alloc_device(&bs.batch_att, 1ll * B * Hq * (c.seq_len + 1) * sizeof(float), 0.f, true); // +1 cho sink
  alloc_device(&bs.batch_tb, 1ll * B * (D * Hq) * sizeof(hip_bfloat16), hip_bfloat16(0.f), true);
  alloc_device(&bs.batch_tb2, 1ll * B * H * sizeof(float), 0.f, true);

  // MLP / MoE
  alloc_device(&bs.batch_router_score, 1ll * B * c.n_experts * sizeof(float), 0.f, true);

  alloc_device(&bs.batch_topk_v, 1ll * B * c.experts_per_token * sizeof(float), 0.f, true);
  alloc_device(&bs.batch_topk_i, 1ll * B * c.experts_per_token * sizeof(int), 0, true);
  
  // Logits buffer (device) + map host con trỏ
  alloc_device(&bs.batch_logits, 1ll * B * c.vocab_size * sizeof(float), 0.f, true);

  if (c.sliding_window > 0) {
    // host-init mask once then copy
    float *hmask = (float*)malloc(1ll*c.seq_len*c.seq_len*sizeof(float));
    for (int i=0;i<c.seq_len;i++) for (int j=0;j<c.seq_len;j++) {
      float v = 0.f;
      if (c.sliding_window > 0 && i - j >= c.sliding_window) v = -INFINITY;
      hmask[i*c.seq_len + j] = v;
    }
    to_device(&bs.mask, hmask, 1ll * c.seq_len * c.seq_len * sizeof(float));
    free(hmask);
  } else {
    bs.mask = nullptr;
  }

  int E = c.n_experts;
  int I = c.intermediate_dim;

  bs.h_counts = (int*)malloc(E * sizeof(int));
  
  alloc_device(&bs.d_counts, E * sizeof(int), 0, true);
  alloc_device(&bs.d_idx_in_batch, E * B * sizeof(int), 0, true);
  
  HIP_CHECK(hipMalloc((void**)&bs.d_in_ptrs, E * B * sizeof(hip_bfloat16*)));
  HIP_CHECK(hipMemset(bs.d_in_ptrs, 0, E * B * sizeof(hip_bfloat16*)));
  
  alloc_device(&bs.batch_mlp1_out, 1ll * B * (2 * I) * sizeof(float), 0.f, true);
  alloc_device(&bs.batch_gate_up, 1ll * B * I * sizeof(hip_bfloat16), hip_bfloat16(0.f), true);
  alloc_device(&bs.d_out, 1ll * E * B * H * sizeof(float), 0.f, true);
}

static void free_batchstate_on_device(BatchState &bs) {
  // Device frees
  free_device(bs.batch_x);
  free_device(bs.batch_t);
  free_device(bs.batch_qkv);
  free_device(bs.batch_q);
  free_device(bs.batch_k);
  free_device(bs.batch_v);
  free_device(bs.batch_att);
  free_device(bs.batch_tb);
  free_device(bs.batch_tb2);
  free_device(bs.batch_router_score);
  free_device(bs.batch_topk_i);
  free_device(bs.batch_topk_v);
  free_device(bs.batch_logits);
  free_device(bs.mask);
  
  free_device(bs.d_counts);
  free_device(bs.d_idx_in_batch);
  free_device(bs.d_out);
  free_device(bs.batch_mlp1_out);
  free_device(bs.batch_gate_up);
  free_device(bs.d_in_ptrs);
  free_host(bs.h_counts);

  // Host frees
  if (bs.prompt_tokens) {
    for (int i = 0; i < MAX_BATCH_SIZE; ++i) {
      free_host(bs.prompt_tokens[i]);
    }

    free_host(bs.prompt_tokens);
  }

  free_host(bs.positions);
  free_host(bs.num_prompt_tokens);
  free_host(bs.current_tokens);
  free_host(bs.finished);
  free_host(bs.req_ids);

}


// -------------------------- Main entry points --------------------------

void memory_map_weights_gpu_120b(TransformerWeights *weights, Config *cfg, float *ptr, BatchState *bs) {
  int head_dim = cfg->head_dim;
  int n_layers = cfg->n_layers;
  int n_experts = cfg->n_experts;

  TransformerWeights* w = &weights[0];
  BatchState* g_batch_state = &bs[0];

  long long size_tmp = max3(
    1ll * cfg->vocab_size * cfg->hidden_dim,
    1ll * n_layers * cfg->hidden_dim * (head_dim * (cfg->n_attn_heads + 2 * cfg->n_kv_heads)),
    1ll * n_layers * n_experts * 2 * cfg->intermediate_dim * cfg->hidden_dim
  );
  hip_bfloat16 *tmp = (hip_bfloat16*)malloc(size_tmp * sizeof(hip_bfloat16));
  if (!tmp) {
      fprintf(stderr, "OOM: cannot allocate host FP16 buffer for tmp\n");
      exit(1);
  }

  convert_fp32_to_bf16_host(ptr, tmp, 1ll * cfg->vocab_size * cfg->hidden_dim);
  to_device(&g_batch_state->d_w_token_embedding_table_bf16, tmp, 1ll * cfg->vocab_size * cfg->hidden_dim * sizeof(hip_bfloat16));
  ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;

  convert_fp32_to_bf16_host_and_transpose(ptr, tmp, 1, cfg->vocab_size, cfg->hidden_dim);
  to_device(&g_batch_state->d_w_out_bf16, tmp, 1ll * cfg->vocab_size * cfg->hidden_dim * sizeof(hip_bfloat16));
  ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;

  to_device(&w->rms_attn_w, ptr, 1ll * n_layers * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim;
  to_device(&w->rms_ffn_w, ptr, 1ll * n_layers * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim;
  to_device(&w->rms_out_w, ptr, 1ll * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * cfg->hidden_dim;

  convert_fp32_to_bf16_host_and_transpose(
    ptr, tmp, n_layers,
    head_dim * (cfg->n_attn_heads + 2 * cfg->n_kv_heads), 
    cfg->hidden_dim
  );
  to_device(
    &g_batch_state->d_w_qkv_bf16, tmp, 
    1ll * n_layers * cfg->hidden_dim * head_dim *
    (cfg->n_attn_heads + 2 * cfg->n_kv_heads) * sizeof(hip_bfloat16)
  );
  ptr += 1ll * n_layers * cfg->hidden_dim *
         (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);

  to_device(&w->b_qkv, ptr,
            1ll * n_layers * (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads) *
            sizeof(float));
  ptr += 1ll * n_layers *
         (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);

  convert_fp32_to_bf16_host_and_transpose(
    ptr, tmp, n_layers,
    cfg->hidden_dim,
    head_dim * cfg->n_attn_heads
  );
  to_device(
    &g_batch_state->d_w_o_bf16, tmp,
    1ll * n_layers * (head_dim * cfg->n_attn_heads) * cfg->hidden_dim * sizeof(hip_bfloat16)
  );
  ptr += 1ll * n_layers * (head_dim * cfg->n_attn_heads) * cfg->hidden_dim;

  to_device(&w->b_o, ptr, 1ll * n_layers * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim;
  to_device(&w->attn_sinks, ptr, 1ll * n_layers * cfg->n_attn_heads * sizeof(float));
  ptr += 1ll * n_layers * cfg->n_attn_heads;

  convert_fp32_to_bf16_host_and_transpose(ptr, tmp, n_layers, n_experts, cfg->hidden_dim);
  to_device(&g_batch_state->d_w_router_bf16, tmp, 1ll * n_layers * n_experts * cfg->hidden_dim * sizeof(hip_bfloat16));
  ptr += 1ll * n_layers * cfg->hidden_dim * n_experts;
  
  to_device(&w->b_router, ptr, 1ll * n_layers * n_experts * sizeof(float));
  ptr += 1ll * n_layers * n_experts;
  
  long long mlp1_per = 2ll * cfg->intermediate_dim * cfg->hidden_dim;

  printf("Converting and copying MLP1 weights...\n");
  fflush(stdout);
  convert_fp32_to_bf16_host_and_transpose(ptr, tmp, n_layers * n_experts, 2 * cfg->intermediate_dim, cfg->hidden_dim);
  ptr += 1ll * n_layers * n_experts * mlp1_per;

  #pragma omp parallel for
  for (int i = 1; i < NUM_GPUS; ++i) {
    HIP_CHECK(hipSetDevice(i));

    printf("GPU %d: Copying MLP1 weights...\n", i);
    fflush(stdout);

    TransformerWeights* w = &weights[i];
    BatchState* g_batch_state = &bs[i];

    int div = n_experts / (NUM_GPUS - 1);
    int mod = n_experts % (NUM_GPUS - 1);
    int num_experts = div + (i <= mod);
    int st_idx = (i - 1) * div + min(i - 1, mod);
    
    long long sz_w = 1ll * num_experts * mlp1_per;
    long long sz_b = 1ll * num_experts * 2 * cfg->intermediate_dim;

    HIP_CHECK(hipMalloc((void**)&g_batch_state->d_w_mlp1_bf16, sz_w * n_layers * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc((void**)&w->b_mlp1, sz_b * n_layers * sizeof(float)));

    for (int j = 0; j < n_layers; ++j) {
      HIP_CHECK(hipMemcpy(
        g_batch_state->d_w_mlp1_bf16 + 1ll * j * sz_w, 
        tmp + 1ll * (j * n_experts + st_idx) * mlp1_per,
        sz_w * sizeof(hip_bfloat16),
        hipMemcpyHostToDevice
      ));

      HIP_CHECK(hipMemcpy(
        w->b_mlp1 + 1ll * j * sz_b, 
        ptr + 1ll * (j * n_experts + st_idx) * 2 * cfg->intermediate_dim,
        sz_b * sizeof(float),
        hipMemcpyHostToDevice
      ));
    }
  }

  ptr += 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim;

  long long mlp2_per = 1ll * cfg->hidden_dim * cfg->intermediate_dim;

  printf("Converting and copying MLP2 weights...\n");
  fflush(stdout);
  convert_fp32_to_bf16_host_and_transpose(ptr, tmp, n_layers * n_experts, cfg->hidden_dim, cfg->intermediate_dim);
  ptr += 1ll * n_layers * n_experts * mlp2_per;

  #pragma omp parallel for
  for (int i = 1; i < NUM_GPUS; ++i) {
    HIP_CHECK(hipSetDevice(i));

    printf("GPU %d: Copying MLP2 weights...\n", i);
    fflush(stdout);

    TransformerWeights* w = &weights[i];
    BatchState* g_batch_state = &bs[i];

    int div = n_experts / (NUM_GPUS - 1);
    int mod = n_experts % (NUM_GPUS - 1);
    int num_experts = div + (i <= mod);
    int st_idx = (i - 1) * div + min(i - 1, mod);

    long long sz_w = 1ll * num_experts * mlp2_per;
    long long sz_b = 1ll * num_experts * cfg->hidden_dim;

    HIP_CHECK(hipMalloc((void**)&g_batch_state->d_w_mlp2_bf16, sz_w * n_layers * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc((void**)&w->b_mlp2, sz_b * n_layers * sizeof(float)));

    for (int j = 0; j < n_layers; ++j) {
      HIP_CHECK(hipMemcpy(
        g_batch_state->d_w_mlp2_bf16 + 1ll * j * sz_w, 
        tmp + 1ll * (j * n_experts + st_idx) * mlp2_per,
        sz_w * sizeof(hip_bfloat16),
        hipMemcpyHostToDevice
      ));

      HIP_CHECK(hipMemcpy(
        w->b_mlp2 + 1ll * j * sz_b, 
        ptr + 1ll * (j * n_experts + st_idx) * cfg->hidden_dim,
        sz_b * sizeof(float),
        hipMemcpyHostToDevice
      ));
    }
  }

  ptr += 1ll * n_layers * n_experts * cfg->hidden_dim;
  
  free(tmp);
}

void free_weights_gpu_120b(TransformerWeights *weights, BatchState *bs) {
  TransformerWeights w = weights[0];
  BatchState* g_batch_state = &bs[0];

  free_device(g_batch_state->d_w_token_embedding_table_bf16); 
  
  free_device(w.rms_attn_w); 
  free_device(w.rms_ffn_w); 
    
  free_device(g_batch_state->d_w_qkv_bf16);
  free_device(w.b_qkv);

  free_device(g_batch_state->d_w_o_bf16);
  free_device(w.b_o); 
    
  free_device(w.attn_sinks); 
     
  free_device(g_batch_state->d_w_router_bf16);
  free_device(w.b_router);

  free_device(w.rms_out_w); 
  free_device(g_batch_state->d_w_out_bf16);
    
  #pragma omp parallel for
  for (int i = 1; i < NUM_GPUS; ++i) {
    HIP_CHECK(hipSetDevice(i));

    TransformerWeights w = weights[i];
    BatchState* g_batch_state = &bs[i];

    free_device(g_batch_state->d_w_mlp1_bf16);
    free_device(w.b_mlp1); 

    free_device(g_batch_state->d_w_mlp2_bf16);
    free_device(w.b_mlp2); 
  }
}

static void alloc_batchstate_on_device_120b(BatchState &bs, const Config &c) {
  // ====== Host-side per-request arrays ======
  bs.positions         = (int  *)calloc(MAX_BATCH_SIZE, sizeof(int));
  bs.num_prompt_tokens = (int  *)calloc(MAX_BATCH_SIZE, sizeof(int));
  bs.prompt_tokens     = (int **)calloc(MAX_BATCH_SIZE, sizeof(int*)); // các phần tử sẽ malloc/free trong inference
  bs.current_tokens    = (int  *)calloc(MAX_BATCH_SIZE, sizeof(int));
  bs.finished          = (bool *)calloc(MAX_BATCH_SIZE, sizeof(bool));
  bs.req_ids           = (int  *)calloc(MAX_BATCH_SIZE, sizeof(int));

  // ====== Device-side batched buffers ======
  const int H   = c.hidden_dim;
  const int D   = c.head_dim;
  const int Hq  = c.n_attn_heads;
  const int Hkv = c.n_kv_heads;
  const int kv_dim = D * Hkv;
  const int qkv_tot = D * (Hq + 2 * Hkv);
  const int B = MAX_BATCH_SIZE;

  // Embedding/residual stream
  alloc_device(&bs.batch_x, 1ll * B * H * sizeof(float), 0.f, true);
  alloc_device(&bs.batch_t, 1ll * B * H * sizeof(hip_bfloat16), hip_bfloat16(0.f), true);

  // Attention projections
  alloc_device(&bs.batch_qkv, 1ll * B * qkv_tot * sizeof(float), 0.f, true);
  alloc_device(&bs.batch_q, 1ll * B * (D * Hq) * sizeof(float), 0.f, true);

  {
    const long long cache_elems = 1ll * c.n_layers / 2 * B * (c.seq_len + c.sliding_window) * kv_dim;
    alloc_device(&bs.batch_k, cache_elems * sizeof(hip_bfloat16), hip_bfloat16(0.f), true);
    alloc_device(&bs.batch_v, cache_elems * sizeof(hip_bfloat16), hip_bfloat16(0.f), true);
  }

  // Attention scores & outputs
  alloc_device(&bs.batch_att, 1ll * B * Hq * (c.seq_len + 1) * sizeof(float), 0.f, true); // +1 cho sink
  alloc_device(&bs.batch_tb, 1ll * B * (D * Hq) * sizeof(hip_bfloat16), hip_bfloat16(0.f), true);
  alloc_device(&bs.batch_tb2, 1ll * B * H * sizeof(float), 0.f, true);

  // MLP / MoE
  alloc_device(&bs.batch_router_score, 1ll * B * c.n_experts * sizeof(float), 0.f, true);

  alloc_device(&bs.batch_topk_v, 1ll * B * c.experts_per_token * sizeof(float), 0.f, true);
  alloc_device(&bs.batch_topk_i, 1ll * B * c.experts_per_token * sizeof(int), 0, true);

  // Logits buffer (device) + map host con trỏ
  alloc_device(&bs.batch_logits, 1ll * B * c.vocab_size * sizeof(float), 0.f, true);

  if (c.sliding_window > 0) {
    // host-init mask once then copy
    float *hmask = (float*)malloc(1ll*c.seq_len*c.seq_len*sizeof(float));
    for (int i=0;i<c.seq_len;i++) for (int j=0;j<c.seq_len;j++) {
      float v = 0.f;
      if (c.sliding_window > 0 && i - j >= c.sliding_window) v = -INFINITY;
      hmask[i*c.seq_len + j] = v;
    }
    to_device(&bs.mask, hmask, 1ll * c.seq_len * c.seq_len * sizeof(float));
    free(hmask);
  } else {
    bs.mask = nullptr;
  }

  int E = c.n_experts;
  int I = c.intermediate_dim;

  bs.h_counts = (int*)malloc(E * sizeof(int));
  
  alloc_device(&bs.d_counts, E * sizeof(int), 0, true);
  alloc_device(&bs.d_idx_in_batch, E * B * sizeof(int), 0, true);
  alloc_device(&bs.d_out, 1ll * E * B * H * sizeof(float), 0.f, true);

  bs.h_in_ptrs = (hip_bfloat16**)malloc(E * B * sizeof(hip_bfloat16*));
  
  HIP_CHECK(hipMalloc((void**)&bs.d_in_ptrs, E * B * sizeof(hip_bfloat16*)));
  HIP_CHECK(hipMemset(bs.d_in_ptrs, 0, E * B * sizeof(hip_bfloat16*)));  

  alloc_device(&bs.d_current_tokens, B * sizeof(int), 0, true);
  alloc_device(&bs.d_positions, B * sizeof(int), 0, true);
  
  const int half = D / 2;
  alloc_device(&bs.cosB, B * half * sizeof(float), 0.f, true);
  alloc_device(&bs.sinB, B * half * sizeof(float), 0.f, true);

  bs.next_tokens = (int*)malloc(B * sizeof(int));
  alloc_device(&bs.d_next_tokens, B * sizeof(int), 0, true);

  bs.coins = (float*)malloc(B * sizeof(float));
  alloc_device(&bs.d_coins, B * sizeof(float), 0.f, true);

  alloc_device(&bs.g_inv_freq_dev, half * sizeof(float), 0.f, true);
  alloc_device(&bs.d_conc, sizeof(float), 0.f, true);
}

void free_batchstate_on_device_120b(BatchState &bs) {
  free_device(bs.batch_x);
  free_device(bs.batch_t);
  free_device(bs.batch_qkv);
  free_device(bs.batch_q);
  free_device(bs.batch_k);
  free_device(bs.batch_v);
  free_device(bs.batch_att);
  free_device(bs.batch_tb);
  free_device(bs.batch_tb2);
  free_device(bs.batch_router_score);
  free_device(bs.batch_topk_i);
  free_device(bs.batch_topk_v);
  free_device(bs.batch_logits);
  free_device(bs.mask);
  
  free_host(bs.h_counts);
  free_device(bs.d_counts);
  
  free_device(bs.d_idx_in_batch);
  free_device(bs.d_out);

  free_host(bs.h_in_ptrs);
  free_device(bs.d_in_ptrs);
  
  free_device(bs.d_current_tokens);
  free_device(bs.d_positions);
  free_device(bs.cosB);
  free_device(bs.sinB);

  free_host(bs.next_tokens);
  free_device(bs.d_next_tokens);

  free_host(bs.coins);
  free_device(bs.d_coins);

  free_device(bs.g_inv_freq_dev);
  free_device(bs.d_conc);

  if (bs.prompt_tokens) {
    for (int i = 0; i < MAX_BATCH_SIZE; ++i) {
      free_host(bs.prompt_tokens[i]);
    }

    free_host(bs.prompt_tokens);
  }

  free_host(bs.positions);
  free_host(bs.num_prompt_tokens);
  free_host(bs.current_tokens);
  free_host(bs.finished);
  free_host(bs.req_ids);
}

void warm_up_120b(Transformer *transformer, Tokenizer *tokenizer) {
  assert(NUM_GPUS == 8);
  MAX_BATCH_SIZE = 1024;
  HIP_CHECK(hipSetDevice(0));
  
  Config &c = transformer->config;
  c.seq_len = 1025;

  batch_states = (BatchState*)malloc(NUM_GPUS * sizeof(BatchState));
  transformer_weights = (TransformerWeights*)malloc(NUM_GPUS * sizeof(TransformerWeights));

  alloc_batchstate_on_device_120b(batch_states[0], c);

  int B = MAX_BATCH_SIZE;
  int H = c.hidden_dim;
  int I = c.intermediate_dim;

  #pragma omp parallel for
  for (int i = 1; i < NUM_GPUS; ++i) {
    HIP_CHECK(hipSetDevice(i));
    BatchState *g_batch_state = &batch_states[i];
    alloc_device(&g_batch_state->batch_x, 1ll * B * H * sizeof(float), 0.f, true);
    alloc_device(&g_batch_state->batch_t, 1ll * B * H * sizeof(hip_bfloat16), hip_bfloat16(0.f), true);
    alloc_device(&g_batch_state->batch_mlp1_out, 1ll * B * (2 * I) * sizeof(float), 0.f, true);
    alloc_device(&g_batch_state->batch_gate_up, 1ll * B * I * sizeof(hip_bfloat16), hip_bfloat16(0.f), true);
  }

  HIP_CHECK(hipSetDevice(0));
  float *weights_ptr = transformer->data + sizeof(c) / sizeof(float);
  memory_map_weights_gpu_120b(transformer_weights, &c, weights_ptr, batch_states);

  HIP_CHECK(hipSetDevice(0));
}

void finish_120b(Transformer *transformer, Tokenizer *tokenizer) {
  HIP_CHECK(hipSetDevice(0));

  free_weights_gpu_120b(transformer_weights, batch_states);
  free(transformer_weights);
  transformer_weights = NULL;

  HIP_CHECK(hipSetDevice(0));
  free_batchstate_on_device_120b(batch_states[0]);

  #pragma omp parallel for
  for (int i = 1; i < NUM_GPUS; ++i) {
    HIP_CHECK(hipSetDevice(i));
    free_device(batch_states[i].batch_x);
    free_device(batch_states[i].batch_t);
    free_device(batch_states[i].batch_mlp1_out);
    free_device(batch_states[i].batch_gate_up);
  }

  free(batch_states); 
  batch_states = NULL;

  HIP_CHECK(hipSetDevice(0));
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the warm-up process
  // TODO:
  // - Memory allocation
  // - Load model
  // - ...

  if (is_120b_model(&transformer->config)) {
    warm_up_120b(transformer, tokenizer);
    return;
  }
  
  Config &c = transformer->config;
  c.seq_len = 1025;
  
  batch_states = (BatchState*)malloc(NUM_GPUS * sizeof(BatchState));
  transformer_weights = (TransformerWeights*)malloc(NUM_GPUS * sizeof(TransformerWeights));

  for (int i = 0; i < NUM_GPUS; ++i) {
    HIP_CHECK(hipSetDevice(i));
    BatchState *g_batch_state = &batch_states[i];
    alloc_batchstate_on_device(*g_batch_state, c);

    float *weights_ptr = transformer->data + sizeof(c) / sizeof(float);
    memory_map_weights_gpu(&transformer_weights[i], &c, weights_ptr, g_batch_state);

    alloc_device(&g_batch_state->d_current_tokens, MAX_BATCH_SIZE * sizeof(int), 0, true);
    alloc_device(&g_batch_state->d_positions, MAX_BATCH_SIZE * sizeof(int), 0, true);
    
    const int half = c.head_dim / 2;
    alloc_device(&g_batch_state->cosB, MAX_BATCH_SIZE * half * sizeof(float), 0.f, true);
    alloc_device(&g_batch_state->sinB, MAX_BATCH_SIZE * half * sizeof(float), 0.f, true);

    g_batch_state->next_tokens = (int*)malloc(MAX_BATCH_SIZE * sizeof(int));
    alloc_device(&g_batch_state->d_next_tokens, MAX_BATCH_SIZE * sizeof(int), 0, true);

    g_batch_state->coins = (float*)malloc(MAX_BATCH_SIZE * sizeof(float));
    alloc_device(&g_batch_state->d_coins, MAX_BATCH_SIZE * sizeof(float), 0.f, true);

    alloc_device(&g_batch_state->g_inv_freq_dev, half * sizeof(float), 0.f, true);
    alloc_device(&g_batch_state->d_conc, sizeof(float), 0.f, true);
  }
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the finish process
  // TODO:
  // - Memory deallocation
  // - Unload model
  // - ...

  if (is_120b_model(&transformer->config)) {
    finish_120b(transformer, tokenizer);
    return;
  }

  #pragma omp parallel for
  for (int i = 0; i < NUM_GPUS; ++i) {
    HIP_CHECK(hipSetDevice(i));
    BatchState *g_batch_state = &batch_states[i];

    free_weights_gpu(transformer_weights[i], g_batch_state);

    free_batchstate_on_device(*g_batch_state);

    free_device(g_batch_state->d_current_tokens);
    free_device(g_batch_state->d_positions);
    free_device(g_batch_state->cosB);
    free_device(g_batch_state->sinB);

    free_host(g_batch_state->next_tokens);
    free_device(g_batch_state->d_next_tokens);

    free_host(g_batch_state->coins);
    free_device(g_batch_state->d_coins);

    free_device(g_batch_state->g_inv_freq_dev);
    free_device(g_batch_state->d_conc);
  }

  free(transformer_weights);
  transformer_weights = NULL;

  free(batch_states); 
  batch_states = NULL;
}

long long inference_120b(Transformer *transformer, Tokenizer *tokenizer,
                         Sampler *sampler, Requests *requests) {

  HIP_CHECK(hipSetDevice(0));
  long long num_token_out = 0;

  int num_reqs = requests->num_reqs;
  const int max_steps = requests->max_seq_len;
  const int batch_size = (num_reqs < MAX_BATCH_SIZE) ? num_reqs : MAX_BATCH_SIZE;

  BatchState *g_batch_state = &batch_states[0];

  for (int i = 0; i < batch_size; ++i) {
    const char *input_seq = get_str_req_ptr(requests, i);

    g_batch_state->prompt_tokens[i] =
        (int*)malloc((strlen(input_seq) + 3) * sizeof(int));
    g_batch_state->num_prompt_tokens[i] = 0;

    encode(tokenizer, input_seq, -1, -1,
            g_batch_state->prompt_tokens[i],
            &g_batch_state->num_prompt_tokens[i],
            transformer->config.initial_context_length);

    g_batch_state->positions[i] = 0;
    g_batch_state->current_tokens[i] = g_batch_state->prompt_tokens[i][0];
    g_batch_state->finished[i] = false;
    g_batch_state->req_ids[i] = i;
  }

  int active_count = batch_size;
  int req_it = batch_size;

  while (active_count) {
    forward_batch_120b(transformer, batch_size);

    bool called_sample = false;
    for (int i = 0; i < batch_size; ++i) {
      if (g_batch_state->finished[i]) {
        continue;
      }
      
      int pos     = ++g_batch_state->positions[i];
      int req_idx = g_batch_state->req_ids[i];

      int next_token;
      if (pos < g_batch_state->num_prompt_tokens[i]) {
        next_token = g_batch_state->prompt_tokens[i][pos];
      } else {
        if (!called_sample) {
          sample_gpu_batch(
            sampler, g_batch_state->batch_logits, batch_size,       
            g_batch_state->d_next_tokens, 
            g_batch_state->coins, g_batch_state->d_coins    
          );
          hipMemcpy(g_batch_state->next_tokens, g_batch_state->d_next_tokens,
                    batch_size * sizeof(int), hipMemcpyDeviceToHost);

          called_sample = true;
        }
        next_token = g_batch_state->next_tokens[i];

        int *output_tokens = get_tok_gen_ptr(requests, req_idx);
        int out_pos = pos - g_batch_state->num_prompt_tokens[i];
        output_tokens[out_pos] = next_token;
      }

      // const char *piece = decode_piece(tokenizer, g_batch_state->current_tokens[i], next_token);
      // safe_printf(piece);
      // fflush(stdout);

      g_batch_state->current_tokens[i] = next_token;

      if (next_token == 199999 || next_token == 200002 || pos + 1 >= max_steps) {
        g_batch_state->finished[i] = true;
        g_batch_state->positions[i] = 0;
        g_batch_state->current_tokens[i] = g_batch_state->prompt_tokens[i][0];
        free(g_batch_state->prompt_tokens[i]);
        g_batch_state->prompt_tokens[i] = nullptr;
        active_count--;

        int *output_tokens = get_tok_gen_ptr(requests, req_idx);
        int out_len = pos - g_batch_state->num_prompt_tokens[i] + 1;
        assert(out_len >= 0);
        output_tokens[out_len] = -1;

        num_token_out += out_len;

        if (req_it < num_reqs) {
          // load new request
          const char *input_seq = get_str_req_ptr(requests, req_it);

          g_batch_state->prompt_tokens[i] =
              (int*)malloc((strlen(input_seq) + 3) * sizeof(int));
          g_batch_state->num_prompt_tokens[i] = 0;

          encode(tokenizer, input_seq, -1, -1,
                  g_batch_state->prompt_tokens[i],
                  &g_batch_state->num_prompt_tokens[i],
                  transformer->config.initial_context_length);

          g_batch_state->positions[i] = 0;
          g_batch_state->current_tokens[i] = g_batch_state->prompt_tokens[i][0];
          g_batch_state->finished[i] = false;
          g_batch_state->req_ids[i] = req_it;

          req_it++;
          active_count++;
        }
      }
    }
  }

  return num_token_out;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
                      
  if (is_120b_model(&transformer->config)) {
    return inference_120b(transformer, tokenizer, sampler, requests);
  }

  long long num_token_out = 0;
  int div = requests->num_reqs / NUM_GPUS;
  int mod = requests->num_reqs % NUM_GPUS;
  
  #pragma omp parallel for
  for (int d = 0; d < NUM_GPUS; d++) {
    HIP_CHECK(hipSetDevice(d));

    BatchState *g_batch_state = &batch_states[d];
    long long num_token_out_local = 0;
  
    const int num_reqs = div + (mod > d);
    assert(num_reqs > 0);

    int start_req = div * d + (d < mod ? d : mod);
    
    const int max_steps = requests->max_seq_len;
    const int batch_size = (num_reqs < MAX_BATCH_SIZE) ? num_reqs : MAX_BATCH_SIZE;
    
    for (int i = 0; i < batch_size; ++i) {
      const char *input_seq = get_str_req_ptr(requests, start_req + i);

      g_batch_state->prompt_tokens[i] =
          (int*)malloc((strlen(input_seq) + 3) * sizeof(int));
      g_batch_state->num_prompt_tokens[i] = 0;

      encode(tokenizer, input_seq, -1, -1,
              g_batch_state->prompt_tokens[i],
              &g_batch_state->num_prompt_tokens[i],
              transformer->config.initial_context_length);

      g_batch_state->positions[i] = 0;
      g_batch_state->current_tokens[i] = g_batch_state->prompt_tokens[i][0];
      g_batch_state->finished[i] = false;
      g_batch_state->req_ids[i] = start_req + i;
    }

    int active_count = batch_size;
    int req_it = batch_size;

    while (active_count) {
      forward_batch(transformer, batch_size);

      bool called_sample = false;
      for (int i = 0; i < batch_size; ++i) {
        if (g_batch_state->finished[i]) {
          continue;
        }
        
        int pos     = ++g_batch_state->positions[i];
        int req_idx = g_batch_state->req_ids[i];

        int next_token;
        if (pos < g_batch_state->num_prompt_tokens[i]) {
          next_token = g_batch_state->prompt_tokens[i][pos];
        } else {
          if (!called_sample) {
            sample_gpu_batch(
              sampler, g_batch_state->batch_logits, batch_size,       
              g_batch_state->d_next_tokens, 
              g_batch_state->coins, g_batch_state->d_coins    
            );
            hipMemcpy(g_batch_state->next_tokens, g_batch_state->d_next_tokens,
                      batch_size * sizeof(int), hipMemcpyDeviceToHost);

            called_sample = true;
          }
          next_token = g_batch_state->next_tokens[i];

          int *output_tokens = get_tok_gen_ptr(requests, req_idx);
          int out_pos = pos - g_batch_state->num_prompt_tokens[i];
          output_tokens[out_pos] = next_token;
        }

        // const char *piece = decode_piece(tokenizer, g_batch_state->current_tokens[i], next_token);
        // safe_printf(piece);
        // fflush(stdout);

        g_batch_state->current_tokens[i] = next_token;

        if (next_token == 199999 || next_token == 200002 || pos + 1 >= max_steps) {
          g_batch_state->finished[i] = true;
          g_batch_state->positions[i] = 0;
          g_batch_state->current_tokens[i] = g_batch_state->prompt_tokens[i][0];
          free(g_batch_state->prompt_tokens[i]);
          g_batch_state->prompt_tokens[i] = nullptr;
          active_count--;

          int *output_tokens = get_tok_gen_ptr(requests, req_idx);
          int out_len = pos - g_batch_state->num_prompt_tokens[i] + 1;
          assert(out_len >= 0);
          output_tokens[out_len] = -1;

          num_token_out_local += out_len;

          if (req_it < num_reqs) {
            // load new request
            const char *input_seq = get_str_req_ptr(requests, start_req + req_it);

            g_batch_state->prompt_tokens[i] =
                (int*)malloc((strlen(input_seq) + 3) * sizeof(int));
            g_batch_state->num_prompt_tokens[i] = 0;

            encode(tokenizer, input_seq, -1, -1,
                    g_batch_state->prompt_tokens[i],
                    &g_batch_state->num_prompt_tokens[i],
                    transformer->config.initial_context_length);

            g_batch_state->positions[i] = 0;
            g_batch_state->current_tokens[i] = g_batch_state->prompt_tokens[i][0];
            g_batch_state->finished[i] = false;
            g_batch_state->req_ids[i] = start_req + req_it;

            req_it++;
            active_count++;
          }
        }
      }

    }

    #pragma omp atomic
    num_token_out += num_token_out_local;
  }

  

  return num_token_out;
}

#endif // GETP_RUN
