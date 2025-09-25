#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <omp.h>

#include "getp_eval.cpp"
#include "../utils.cpp"

#ifndef GETP_RUN
#define GETP_RUN

typedef struct {

    float* batch_t;              // [max_batch_size, hidden_dim]
    float** batch_t_gap;

    float* batch_mlp1_out;       // [max_batch_size, 2 * intermediate_dim]
    float* batch_gate;           // [max_batch_size, intermediate_dim]
    float* batch_up;             // [max_batch_size, intermediate_dim]
    float* batch_gate_up;        // [max_batch_size, intermediate_dim]

    float* batch_wexps;
    float* host_wexps;

    int* idx_in_batch;
    int num_batch;

} BatchStateMOE;

typedef struct {
    // ==== Per-request state (host) ====
    int* positions;              // [batch_size]  vị trí hiện tại của từng sequence
    int* num_prompt_tokens;      // [batch_size]  độ dài prompt của từng sequence
    int** prompt_tokens;         // [batch_size][] mảng token prompt (malloc/free trên host)
    int* current_tokens;         // [batch_size]  token hiện tại của từng sequence
    bool* finished;              // [batch_size]  cờ kết thúc cho từng sequence
    int* req_ids;

    // Logits
    float** logits_batch;        // [batch_size]  (tùy chọn) mảng con trỏ tới logits device per-seq
    float*  batch_logits;        // [batch_size, vocab_size] logits trên device (contiguous)

    // ==== Pre-allocated buffers for batched operations (device) ====
    // Kích thước sử dụng: [max_batch_size, ...]; bạn cấp phát theo max_batch_size, dùng batch_size phần đầu.

    // Emb/Residual stream
    float* batch_x;              // [max_batch_size, hidden_dim]
    float* batch_t;              // [max_batch_size, hidden_dim]

    // Attention projections
    float* batch_qkv;            // [max_batch_size, head_dim * (n_attn_heads + 2*n_kv_heads)]
    float* batch_q;              // [max_batch_size, head_dim * n_attn_heads]

    // Attention caches (cho toàn bộ lịch sử 0..pos, theo layer & batch)
    // Layout gợi ý: [n_layers, max_batch_size, seq_len, kv_dim]
    float* batch_k;              // size = n_layers * max_batch_size * seq_len * (head_dim * n_kv_heads)
    float* batch_v;              // như trên

    // Attention scores & outputs
    float* batch_att;            // [max_batch_size, n_attn_heads, (seq_len + 1)]  // +1 để append sink
    float* batch_tb;             // [max_batch_size, head_dim * n_attn_heads]
    float* batch_tb2;            // [max_batch_size, hidden_dim]

    // ==== MLP / MoE buffers (device) ====
    float* batch_router_score;   // [max_batch_size, n_experts]
    float* batch_topk_v;         // [max_batch_size, experts_per_token]
    int*   batch_topk_i;         // [max_batch_size, experts_per_token]

    float* mask;

    int* h_topk_i;
    float* h_topk_v;

    int* d_current_tokens;
    int* d_positions;
    float* cosB;
    float* sinB;

    float* h_p;

    hip_bfloat16* d_w_mlp1_bf16;
    hip_bfloat16* d_w_mlp2_bf16;
    
    // ======= Bộ nhớ cache inv_freq (tạo 1 lần, dùng lại) =======
    float* g_inv_freq_dev; // [half] on device
    float* d_conc;
    int    g_cached_hd    = 0;
    float  g_cached_base  = 0.f, g_cached_scale = 0.f, g_cached_ic = 0.f, g_cached_b = 0.f, g_cached_a = 0.f;
    float  g_concentration = 1.f;

    BatchStateMOE* MOE_tmp_batch_state;

} BatchState;

int NUM_GPUS = 4;
const int MAX_BATCH_SIZE = 128;
BatchState* batch_states = NULL;
TransformerWeights* transformer_weights = NULL;

#include "../forward.cpp"
#include "../sample.cpp"

// ------------------------ GPU allocations / deallocations ------------------------

static void to_device(float **dptr, const float *hptr, size_t nbytes) {
  HIP_CHECK(hipMalloc((void**)dptr, nbytes));
  HIP_CHECK(hipMemcpy(*dptr, hptr, nbytes, hipMemcpyHostToDevice));
}

static void to_device_and_transpose(float **dptr, const float *hptr, int n, int rows, int cols) {
  long long total_elems = 1ll * n * rows * cols;
  HIP_CHECK(hipMalloc((void**)dptr, total_elems * sizeof(float)));
  float *tmp = (float*)malloc(total_elems * sizeof(float));

  // transpose
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    const float *s = hptr + (long long)i * rows * cols;
    float *d = tmp + (long long)i * rows * cols;
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        d[c * rows + r] = s[r * cols + c];
      }
    }
  }

  HIP_CHECK(hipMemcpy(*dptr, tmp, total_elems * sizeof(float), hipMemcpyHostToDevice));
  free(tmp);
}

static void alloc_device(float **dptr, size_t nbytes, float fill=0.f, bool do_set=false) {
  HIP_CHECK(hipMalloc((void**)dptr, nbytes));
  if (do_set) {
    int n = (int)(nbytes / sizeof(float));
    set(*dptr, fill, n);
  }
}
  
void free_float_device(float *&p) {
  if (p) {
    hipFree(p);
    p = nullptr; 
  } 
}

void free_int_device(int *&p) {
  if (p) {
    hipFree(p);
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
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        d[c * rows + r] = hip_bfloat16(s[r * cols + c]);
      }
    }
  }
}

void memory_map_weights_gpu(TransformerWeights *w, Config *cfg, float *ptr, BatchState *g_batch_state) {
  int head_dim = cfg->head_dim;
  int n_layers = cfg->n_layers;
  int n_experts = cfg->n_experts;

  to_device(&w->token_embedding_table, ptr, 1ll*cfg->vocab_size*cfg->hidden_dim*sizeof(float));
  ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;

  // to_device(&w->out, ptr, 1ll*cfg->vocab_size*cfg->hidden_dim*sizeof(float));
  to_device_and_transpose(&w->out, ptr, 1, cfg->vocab_size, cfg->hidden_dim);
  ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;

  to_device(&w->rms_attn_w, ptr, 1ll * n_layers * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim;
  to_device(&w->rms_ffn_w, ptr, 1ll * n_layers * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim;
  to_device(&w->rms_out_w, ptr, 1ll * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * cfg->hidden_dim;
  // hey it's qkvqkv, not qqkkvv
  // to_device(&w->w_qkv, ptr,
  //           1ll * n_layers * cfg->hidden_dim *
  //           (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads) *
  //           sizeof(float));
  to_device_and_transpose(&w->w_qkv, ptr, n_layers,
                          head_dim * (cfg->n_attn_heads + 2 * cfg->n_kv_heads), 
                          cfg->hidden_dim);
  ptr += 1ll * n_layers * cfg->hidden_dim *
         (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);

  to_device(&w->b_qkv, ptr,
            1ll * n_layers * (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads) *
            sizeof(float));
  ptr += 1ll * n_layers *
         (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);

  // to_device(&w->w_o, ptr,
  //           1ll * n_layers * (head_dim * cfg->n_attn_heads) * cfg->hidden_dim *
  //           sizeof(float));
  to_device_and_transpose(&w->w_o, ptr, n_layers,
                          cfg->hidden_dim,
                          head_dim * cfg->n_attn_heads);
  ptr += 1ll * n_layers * (head_dim * cfg->n_attn_heads) * cfg->hidden_dim;

  to_device(&w->b_o, ptr, 1ll * n_layers * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim;
  to_device(&w->attn_sinks, ptr, 1ll * n_layers * cfg->n_attn_heads * sizeof(float));
  ptr += 1ll * n_layers * cfg->n_attn_heads;

  // to_device(&w->w_router, ptr, 1ll * n_layers * cfg->hidden_dim * n_experts * sizeof(float));
  to_device_and_transpose(&w->w_router, ptr, n_layers, n_experts, cfg->hidden_dim);
  ptr += 1ll * n_layers * cfg->hidden_dim * n_experts;
  
  to_device(&w->b_router, ptr, 1ll * n_layers * n_experts * sizeof(float));
  ptr += 1ll * n_layers * n_experts;
    
  long long elems_w_mlp1 = 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim * cfg->hidden_dim;
    
  hip_bfloat16 *tmp = (hip_bfloat16*)malloc(elems_w_mlp1 * sizeof(hip_bfloat16));
  if (!tmp) {
      fprintf(stderr, "OOM: cannot allocate host FP16 buffer for w_mlp1\n");
      exit(1);
  }

  // convert_fp32_to_bf16_host(ptr, tmp, elems_w_mlp1);
  convert_fp32_to_bf16_host_and_transpose(ptr, tmp, n_layers * n_experts, 2 * cfg->intermediate_dim, cfg->hidden_dim);

  HIP_CHECK(hipMalloc(&g_batch_state->d_w_mlp1_bf16, elems_w_mlp1 * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemcpy(g_batch_state->d_w_mlp1_bf16, tmp, elems_w_mlp1 * sizeof(hip_bfloat16), hipMemcpyHostToDevice));

  ptr += elems_w_mlp1;
  
  to_device(&w->b_mlp1, ptr, 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim * sizeof(float));
  ptr += 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim;

  long long elems_w_mlp2 = 1ll * n_layers * n_experts * cfg->hidden_dim * cfg->intermediate_dim;

  // convert_fp32_to_bf16_host(ptr, tmp, elems_w_mlp2);
  convert_fp32_to_bf16_host_and_transpose(ptr, tmp, n_layers * n_experts, cfg->hidden_dim, cfg->intermediate_dim);

  HIP_CHECK(hipMalloc(&g_batch_state->d_w_mlp2_bf16, elems_w_mlp2 * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemcpy(g_batch_state->d_w_mlp2_bf16, tmp, elems_w_mlp2 * sizeof(hip_bfloat16), hipMemcpyHostToDevice));

  ptr += elems_w_mlp2;
  
  free(tmp);
  
  to_device(&w->b_mlp2, ptr, 1ll * n_layers * n_experts * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * n_experts * cfg->hidden_dim;
}

static void free_weights_gpu(TransformerWeights &w, BatchState *g_batch_state) {
  free_float_device(w.token_embedding_table); 
  
  free_float_device(w.rms_attn_w); 
  free_float_device(w.rms_ffn_w); 
    
  free_float_device(w.w_qkv); 
  free_float_device(w.b_qkv);

  free_float_device(w.w_o); 
  free_float_device(w.b_o); 
    
  free_float_device(w.attn_sinks); 
    
  free_float_device(w.w_router); 
  free_float_device(w.b_router);
    
  // free_float_device(w.w_mlp1); 
  free_float_device(w.b_mlp1); 
    
  // free_float_device(w.w_mlp2);
  free_float_device(w.b_mlp2); 
    
  free_float_device(w.rms_out_w); 
  free_float_device(w.out);

  if (g_batch_state->d_w_mlp1_bf16) {
    hipFree(g_batch_state->d_w_mlp1_bf16);
    g_batch_state->d_w_mlp1_bf16 = nullptr;
  }

  if (g_batch_state->d_w_mlp2_bf16) {
    hipFree(g_batch_state->d_w_mlp2_bf16);
    g_batch_state->d_w_mlp2_bf16 = nullptr;
  }
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

  // logits: host mảng con trỏ & device buffer chứa logits liên tiếp
  bs.logits_batch = (float**)malloc(MAX_BATCH_SIZE * sizeof(float*));

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
  alloc_device(&bs.batch_t, 1ll * B * H * sizeof(float), 0.f, true);

  // Attention projections
  alloc_device(&bs.batch_qkv, 1ll * B * qkv_tot * sizeof(float), 0.f, true);
  alloc_device(&bs.batch_q, 1ll * B * (D * Hq) * sizeof(float), 0.f, true);

  // Attention caches: [n_layers, MAX_BATCH_SIZE, seq_len, kv_dim]
  {
    const long long cache_elems = 1ll * c.n_layers * B * c.seq_len * kv_dim;
    alloc_device(&bs.batch_k, cache_elems * sizeof(float), 0.f, true);
    alloc_device(&bs.batch_v, cache_elems * sizeof(float), 0.f, true);
  }

  // Attention scores & outputs
  alloc_device(&bs.batch_att, 1ll * B * Hq * (c.seq_len + 1) * sizeof(float), 0.f, true); // +1 cho sink
  alloc_device(&bs.batch_tb, 1ll * B * (D * Hq) * sizeof(float), 0.f, true);
  alloc_device(&bs.batch_tb2, 1ll * B * H * sizeof(float), 0.f, true);

  // MLP / MoE
  alloc_device(&bs.batch_router_score, 1ll * B * c.n_experts * sizeof(float), 0.f, true);

  HIP_CHECK(hipMalloc((void**)&bs.batch_topk_v, 1ll * B * c.experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc((void**)&bs.batch_topk_i, 1ll * B * c.experts_per_token * sizeof(int)));
  
  // Logits buffer (device) + map host con trỏ
  alloc_device(&bs.batch_logits, 1ll * B * c.vocab_size * sizeof(float), 0.f, true);
  for (int i = 0; i < MAX_BATCH_SIZE; ++i) {
    bs.logits_batch[i] = bs.batch_logits + 1ll * i * c.vocab_size;
  }

  if (c.sliding_window > 0) {
    alloc_device(&bs.mask, 1ll*c.seq_len*c.seq_len*sizeof(float), 0.f, true);
    // host-init mask once then copy
    float *hmask = (float*)malloc(1ll*c.seq_len*c.seq_len*sizeof(float));
    for (int i=0;i<c.seq_len;i++) for (int j=0;j<c.seq_len;j++) {
      float v = 0.f;
      if (c.sliding_window > 0 && i - j >= c.sliding_window) v = -INFINITY;
      hmask[i*c.seq_len + j] = v;
    }
    HIP_CHECK(hipMemcpy(bs.mask, hmask, 1ll*c.seq_len*c.seq_len*sizeof(float), hipMemcpyHostToDevice));
    free(hmask);
  } else {
    bs.mask = nullptr;
  }
}

static void free_batchstate_on_device(BatchState &bs) {
  // Device frees
  free_float_device(bs.batch_x);
  free_float_device(bs.batch_t);
  free_float_device(bs.batch_qkv);
  free_float_device(bs.batch_q);
  free_float_device(bs.batch_k);
  free_float_device(bs.batch_v);
  free_float_device(bs.batch_att);
  free_float_device(bs.batch_tb);
  free_float_device(bs.batch_tb2);
  free_float_device(bs.batch_router_score);
  free_int_device(bs.batch_topk_i);
  free_float_device(bs.batch_topk_v);
  free_float_device(bs.batch_logits);
  free_float_device(bs.mask);

  // Host frees
  if (bs.prompt_tokens) {
    for (int i = 0; i < MAX_BATCH_SIZE; ++i) {
      if (bs.prompt_tokens[i]) {
        free(bs.prompt_tokens[i]);
        bs.prompt_tokens[i] = NULL; 
      }
    }

    free(bs.prompt_tokens); 
    bs.prompt_tokens = NULL;
  }

  if (bs.positions) {
    free(bs.positions);
    bs.positions = NULL;
  }

  if (bs.num_prompt_tokens) {
    free(bs.num_prompt_tokens);
    bs.num_prompt_tokens = NULL;
  }

  if (bs.current_tokens) {
    free(bs.current_tokens);
    bs.current_tokens = NULL;
  }

  if (bs.finished) {
    free(bs.finished);
    bs.finished = NULL;
  }

  if (bs.logits_batch) {
    free(bs.logits_batch);
    bs.logits_batch = NULL;
  }

  if (bs.req_ids) {
    free(bs.req_ids); 
    bs.req_ids = NULL;
  }

}

// -------------------------- BatchStateMOE management --------------------------

static void alloc_batchstate_moe_on_device(BatchStateMOE &bs, const Config &c) {
  const int H   = c.hidden_dim;
  const int I   = c.intermediate_dim;
  const int B   = MAX_BATCH_SIZE;

  alloc_device(&bs.batch_t,   1ll * B * H * sizeof(float), 0.f, true);
  HIP_CHECK(hipMalloc(&bs.batch_t_gap, B * sizeof(float*)));

  alloc_device(&bs.batch_mlp1_out,  1ll * B * (2 * I) * sizeof(float), 0.f, true);

  alloc_device(&bs.batch_gate,      1ll * B * I * sizeof(float), 0.f, true);
  alloc_device(&bs.batch_up,        1ll * B * I * sizeof(float), 0.f, true);
  alloc_device(&bs.batch_gate_up,   1ll * B * I * sizeof(float), 0.f, true);

  alloc_device(&bs.batch_wexps, B * sizeof(float), 0.f, true);

  bs.host_wexps = (float*)malloc(B * sizeof(float));

  bs.idx_in_batch = (int*)malloc(B * sizeof(int));
}

static void free_batchstate_moe_on_device(BatchStateMOE &bs) {
  free_float_device(bs.batch_t);
  if (bs.batch_t_gap) {
    hipFree(bs.batch_t_gap);
    bs.batch_t_gap = nullptr;
  }

  free_float_device(bs.batch_mlp1_out);

  free_float_device(bs.batch_gate);
  free_float_device(bs.batch_up);
  free_float_device(bs.batch_gate_up);

  free_float_device(bs.batch_wexps);

  if (bs.host_wexps) {
    free(bs.host_wexps);
    bs.host_wexps = nullptr;
  }

  if (bs.idx_in_batch) {
    free(bs.idx_in_batch);
    bs.idx_in_batch = nullptr;
  }
}

// -------------------------- Main entry points --------------------------

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the warm-up process
  // TODO:
  // - Memory allocation
  // - Load model
  // - ...
  
  Config &c = transformer->config;
  c.seq_len /= 2;
  
  batch_states = (BatchState*)malloc(NUM_GPUS * sizeof(BatchState));
  transformer_weights = (TransformerWeights*)malloc(NUM_GPUS * sizeof(TransformerWeights));

  #pragma omp parallel for num_threads(NUM_GPUS)
  for (int i = 0; i < NUM_GPUS; ++i) {
    HIP_CHECK(hipSetDevice(i));
    BatchState *g_batch_state = &batch_states[i];

    float *weights_ptr = transformer->data + sizeof(c) / sizeof(float);
    memory_map_weights_gpu(&transformer_weights[i], &c, weights_ptr, g_batch_state);

    alloc_batchstate_on_device(*g_batch_state, c);

    g_batch_state->h_topk_i = (int*)  malloc(MAX_BATCH_SIZE * c.experts_per_token * sizeof(int));
    g_batch_state->h_topk_v = (float*)malloc(MAX_BATCH_SIZE * c.experts_per_token * sizeof(float));

    HIP_CHECK(hipMalloc(&g_batch_state->d_current_tokens, MAX_BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipMalloc(&g_batch_state->d_positions, MAX_BATCH_SIZE * sizeof(int)));
    
    const int half = c.head_dim / 2;
    HIP_CHECK(hipMalloc(&g_batch_state->cosB, MAX_BATCH_SIZE * half * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_batch_state->sinB, MAX_BATCH_SIZE * half * sizeof(float)));

    g_batch_state->h_p = (float*)malloc(c.vocab_size * sizeof(float));
      
    HIP_CHECK(hipMalloc(&g_batch_state->g_inv_freq_dev, half * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_batch_state->d_conc, sizeof(float)));

    g_batch_state->MOE_tmp_batch_state = (BatchStateMOE*)malloc(c.n_experts * sizeof(BatchStateMOE));
    for (int i = 0; i < c.n_experts; ++i) {
      alloc_batchstate_moe_on_device(g_batch_state->MOE_tmp_batch_state[i], c);
    }
  }
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the finish process
  // TODO:
  // - Memory deallocation
  // - Unload model
  // - ...
  #pragma omp parallel for num_threads(NUM_GPUS)
  for (int i = 0; i < NUM_GPUS; ++i) {
    HIP_CHECK(hipSetDevice(i));
    BatchState *g_batch_state = &batch_states[i];

    free_weights_gpu(transformer_weights[i], g_batch_state);

    free_batchstate_on_device(*g_batch_state);
    
    if (g_batch_state->h_topk_i) {
      free(g_batch_state->h_topk_i);
      g_batch_state->h_topk_i = nullptr;
    }
    if (g_batch_state->h_topk_v) {
      free(g_batch_state->h_topk_v);
      g_batch_state->h_topk_v = nullptr;
    }

    HIP_CHECK(hipFree(g_batch_state->d_current_tokens));
    HIP_CHECK(hipFree(g_batch_state->d_positions));
    HIP_CHECK(hipFree(g_batch_state->cosB));
    HIP_CHECK(hipFree(g_batch_state->sinB));

    free(g_batch_state->h_p);

    HIP_CHECK(hipFree(g_batch_state->g_inv_freq_dev));
    HIP_CHECK(hipFree(g_batch_state->d_conc));

    for (int i = 0; i < transformer->config.n_experts; ++i) {
      free_batchstate_moe_on_device(g_batch_state->MOE_tmp_batch_state[i]);
    }
    free(g_batch_state->MOE_tmp_batch_state); 
    g_batch_state->MOE_tmp_batch_state = NULL;
  }

  free(transformer_weights);
  transformer_weights = NULL;

  free(batch_states); 
  batch_states = NULL;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  long long num_token_out = 0;
  int div = requests->num_reqs / NUM_GPUS;
  int mod = requests->num_reqs % NUM_GPUS;
  
  #pragma omp parallel for num_threads(NUM_GPUS)
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

          next_token = sample_gpu(sampler, g_batch_state->logits_batch[i]);

          int *output_tokens = get_tok_gen_ptr(requests, req_idx);
          int out_pos = pos - g_batch_state->num_prompt_tokens[i];
          output_tokens[out_pos] = next_token;
        }

        // const char *piece = decode_piece(tokenizer, g_batch_state->current_tokens[i], next_token);
        // safe_printf(piece);
        // fflush(stdout);

        g_batch_state->current_tokens[i] = next_token;

        if (next_token == 199999 || next_token == 200002 || pos >= max_steps) {
          g_batch_state->finished[i] = true;
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
