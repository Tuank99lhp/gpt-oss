// TODO: Modify this file to optimize end-to-end throughput
#include "getp_eval.cpp"
#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include "win.h"
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <omp.h>

#include "../tokenizer.hpp"

#include "../forward.cpp"
#include "../sample.cpp"

#include "../batch_hip/rms_norm.cpp"
#include "../batch_hip/matmul.cpp"
#include "../batch_hip/add_bias.cpp"
#include "../batch_hip/split_qkv.cpp"
#include "../batch_hip/axpy.cpp"
#include "../batch_hip/attention.cpp"
#include "../batch_hip/swiglu.cpp"
#include "../batch_hip/rope.cpp"
#include "../batch_hip/softmax.cpp"
#include "../batch_hip/top_k.cpp"

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
    float* batch_e_agg;          // [max_batch_size, hidden_dim]

    float* batch_mlp1_out;       // [max_batch_size, 2 * intermediate_dim]
    float* batch_gate;           // [max_batch_size, intermediate_dim]
    float* batch_up;             // [max_batch_size, intermediate_dim]
    float* batch_gate_up;        // [max_batch_size, intermediate_dim]

    float* batch_wexps;

    float* mask;
} BatchState;

const int MAX_BATCH_SIZE = 32;
static BatchState* g_batch_state = NULL;

BatchState *MOE_batch_dev_state = nullptr;
float **MOE_batch_partial_on_dev0 = nullptr; 

int   *h_topk_i = nullptr;
float *h_topk_v = nullptr;

// ------------------------------- Helpers ---------------------------------

#define HIP_CHECK(cmd) do { \
  hipError_t e = (cmd);     \
  if (e != hipSuccess) {    \
    fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
            (int)e, hipGetErrorString(e), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

// ------------------------ GPU setup / allocations ------------------------

static void to_device(float **dptr, const float *hptr, size_t nbytes) {
  HIP_CHECK(hipMalloc((void**)dptr, nbytes));
  HIP_CHECK(hipMemcpy(*dptr, hptr, nbytes, hipMemcpyHostToDevice));
}

static void alloc_device(float **dptr, size_t nbytes, float fill=0.f, bool set=false) {
  HIP_CHECK(hipMalloc((void**)dptr, nbytes));
  if (set) {
    int n = (int)(nbytes / sizeof(float));
    int bs = 256, gs = (n + bs - 1) / bs;
    hipLaunchKernelGGL(k_set, dim3(gs), dim3(bs), 0, 0, *dptr, fill, n);
    HIP_CHECK(hipDeviceSynchronize());
  }
}

// -------------------------- File mapping (host) --------------------------

void memory_map_weights_gpu(TransformerWeights *w, Config *cfg, float *ptr) {
  int head_dim = cfg->head_dim;
  int n_layers = cfg->n_layers;
  int n_experts = cfg->n_experts;

  to_device(&w->token_embedding_table, ptr, 1ll*cfg->vocab_size*cfg->hidden_dim*sizeof(float));
  ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;
  to_device(&w->out, ptr, 1ll*cfg->vocab_size*cfg->hidden_dim*sizeof(float));
  ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;
  to_device(&w->rms_attn_w, ptr, 1ll * n_layers * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim;
  to_device(&w->rms_ffn_w, ptr, 1ll * n_layers * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim;
  to_device(&w->rms_out_w, ptr, 1ll * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * cfg->hidden_dim;
  // hey it's qkvqkv, not qqkkvv
  to_device(&w->w_qkv, ptr,
            1ll * n_layers * cfg->hidden_dim *
            (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads) *
            sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim *
         (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
  to_device(&w->b_qkv, ptr,
            1ll * n_layers * (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads) *
            sizeof(float));
  ptr += 1ll * n_layers *
         (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
  to_device(&w->w_o, ptr,
            1ll * n_layers * (head_dim * cfg->n_attn_heads) * cfg->hidden_dim *
            sizeof(float));
  ptr += 1ll * n_layers * (head_dim * cfg->n_attn_heads) * cfg->hidden_dim;
  to_device(&w->b_o, ptr, 1ll * n_layers * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim;
  to_device(&w->attn_sinks, ptr, 1ll * n_layers * cfg->n_attn_heads * sizeof(float));
  ptr += 1ll * n_layers * cfg->n_attn_heads;
  to_device(&w->w_router, ptr, 1ll * n_layers * cfg->hidden_dim * n_experts * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim * n_experts;
  to_device(&w->b_router, ptr, 1ll * n_layers * n_experts * sizeof(float));
  ptr += 1ll * n_layers * n_experts;
  // hey it's gate_upgate_up, not gategateupup
  // to_device(&w->w_mlp1, ptr,
  //           1ll * n_layers * n_experts * cfg->hidden_dim * 2 * cfg->intermediate_dim *
  //           sizeof(float));
  // ptr +=
  //     1ll * n_layers * n_experts * 2 * cfg->intermediate_dim * cfg->hidden_dim;
  // to_device(&w->b_mlp1, ptr, 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim * sizeof(float));
  // ptr += 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim;
  // to_device(&w->w_mlp2, ptr,
  //           1ll * n_layers * n_experts * cfg->hidden_dim * cfg->intermediate_dim *
  //           sizeof(float));
  // ptr += 1ll * n_layers * n_experts * cfg->hidden_dim * cfg->intermediate_dim;
  // to_device(&w->b_mlp2, ptr, 1ll * n_layers * n_experts * cfg->hidden_dim * sizeof(float));
  // ptr += 1ll * n_layers * n_experts * cfg->hidden_dim;
}

void load_checkpoint_gpu(char *ckpt, Config *config, TransformerWeights *weights,
                     int *fd, float **data, ssize_t *file_size) {
  FILE *file = fopen(ckpt, "rb");
  if (!file) {
    fprintf(stderr, "Couldn't open file %s\n", ckpt);
    exit(EXIT_FAILURE);
  }

  // read in the config header
  // load sizeof(Config) bytes into config
  if (fread(config, sizeof(Config), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  // figure out the file size
  // printf("vocab_size: %d\n", config->vocab_size);
  // printf("hidden_dim: %d\n", config->hidden_dim);
  // printf("n_experts: %d\n", config->n_experts);
  // printf("experts_per_token: %d\n", config->experts_per_token);
  // printf("intermediate_dim: %d\n", config->intermediate_dim);
  // printf("n_layers: %d\n", config->n_layers);
  // printf("head_dim: %d\n", config->head_dim);
  // printf("n_attn_heads: %d\n", config->n_attn_heads);
  // printf("n_kv_heads: %d\n", config->n_kv_heads);
  // printf("max_seq_len: %d\n", config->seq_len);
  // printf("init context len: %d\n", config->initial_context_length);
  // printf("rope theta: %f\n", config->rope_theta);
  // printf("rope_scaling_factor: %f\n", config->rope_scaling_factor);
  // printf("sliding window: %d\n", config->sliding_window);
  // printf("swiglu_limit: %f\n", config->swiglu_limit);
  fseek(file, 0, SEEK_END); // move file pointer to end of file

  *file_size = ftell(file); // get the file size, in bytes
  fclose(file);
  // memory map the Transformer weights into the data pointer
  *fd = open(ckpt, O_RDONLY); // open in read only mode
  if (*fd == -1) {
    fprintf(stderr, "open failed\n");
    exit(EXIT_FAILURE);
  }
  *data = reinterpret_cast<float *>(
      mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0));
  if (*data == MAP_FAILED) {
    fprintf(stderr, "mmap failed!\n");
    exit(EXIT_FAILURE);
  }
  float *weights_ptr = *data + sizeof(Config) / sizeof(float);
  memory_map_weights_gpu(weights, config, weights_ptr);
}

static void malloc_state_gpu(Transformer *T) {
  const Config &c = T->config;
  RunState &s = T->state;

  alloc_device(&s.x,        c.hidden_dim*sizeof(float), 0.f, true);
  alloc_device(&s.t,        c.hidden_dim*sizeof(float), 0.f, true);
  alloc_device(&s.tb,       c.head_dim*c.n_attn_heads*sizeof(float), 0.f, true);
  alloc_device(&s.tb2,      c.hidden_dim*sizeof(float), 0.f, true);
  alloc_device(&s.router_score,   c.n_experts*sizeof(float), 0.f, true);
  HIP_CHECK(hipMalloc((void**)&s.topk_v, c.experts_per_token*sizeof(float)));
  HIP_CHECK(hipMalloc((void**)&s.topk_i, c.experts_per_token*sizeof(int)));
  alloc_device(&s.mlp1_out, 2*c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.gate,     c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.up,       c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.gate_up,  c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.e_agg,    c.hidden_dim*sizeof(float), 0.f, true);

  int qkv_tot = c.head_dim*(c.n_attn_heads + 2*c.n_kv_heads);
  alloc_device(&s.qkv,    qkv_tot*sizeof(float), 0.f, true);
  alloc_device(&s.q,      c.head_dim*c.n_attn_heads*sizeof(float), 0.f, true);
  // k_cur/v_cur views are offsets into caches (no alloc here)
  alloc_device(&s.att,    (c.n_attn_heads*(c.seq_len+1))*sizeof(float), 0.f, true);
  alloc_device(&s.logits, c.vocab_size*sizeof(float), 0.f, true);

  int kv_dim = c.head_dim * c.n_kv_heads;
  size_t cache_elems = 1ll*c.n_layers*c.seq_len*kv_dim;
  alloc_device(&s.key_cache,   cache_elems*sizeof(float), 0.f, true);
  alloc_device(&s.value_cache, cache_elems*sizeof(float), 0.f, true);

  if (c.sliding_window > 0) {
    alloc_device(&s.mask, 1ll*c.seq_len*c.seq_len*sizeof(float), 0.f, true);
    // host-init mask once then copy
    float *hmask = (float*)malloc(1ll*c.seq_len*c.seq_len*sizeof(float));
    for (int i=0;i<c.seq_len;i++) for (int j=0;j<c.seq_len;j++) {
      float v = 0.f;
      if (c.sliding_window > 0 && i - j >= c.sliding_window) v = -INFINITY;
      hmask[i*c.seq_len + j] = v;
    }
    HIP_CHECK(hipMemcpy(s.mask, hmask, 1ll*c.seq_len*c.seq_len*sizeof(float), hipMemcpyHostToDevice));
    free(hmask);
  } else {
    s.mask = nullptr;
  }
}

// ---------- Multi-GPU MoE infra (add to getp_run.cpp) ----------
// Place after your existing helpers/kernels, before build_transformer_gpu/forward_gpu.

int MOE_NGPUS = 0;           // number of devices used for expert parallelism
int MOE_GROUP_SIZE = 0;      // experts per device (n_experts / MOE_NGPUS)

// per-device pointers for MoE shards and per-device RunState
float **MOE_dev_w_mlp1 = nullptr;
float **MOE_dev_w_mlp2 = nullptr;
float **MOE_dev_b_mlp1 = nullptr;
float **MOE_dev_b_mlp2 = nullptr;
RunState *MOE_dev_state = nullptr;      // host-side array of RunState (each entry holds device pointers)
float **MOE_partial_on_dev0 = nullptr;   // device-0 pointers for partial results (one per device)

// Helper: allocate RunState fields on a given device (uses alloc_device helper already in file)
// note: call hipSetDevice(dev) before calling this function
static void alloc_runstate_on_device(RunState &s, const Config &c) {
  alloc_device(&s.x,        c.hidden_dim*sizeof(float), 0.f, true);
  alloc_device(&s.t,        c.hidden_dim*sizeof(float), 0.f, true);
  alloc_device(&s.tb,       c.head_dim*c.n_attn_heads*sizeof(float), 0.f, true);
  alloc_device(&s.tb2,      c.hidden_dim*sizeof(float), 0.f, true);
  alloc_device(&s.router_score,   c.n_experts*sizeof(float), 0.f, true);
  HIP_CHECK(hipMalloc((void**)&s.topk_v, c.experts_per_token*sizeof(float)));
  HIP_CHECK(hipMalloc((void**)&s.topk_i, c.experts_per_token*sizeof(int)));
  alloc_device(&s.mlp1_out, 2*c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.gate,     c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.up,       c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.gate_up,  c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.e_agg,    c.hidden_dim*sizeof(float), 0.f, true);

  int qkv_tot = c.head_dim*(c.n_attn_heads + 2*c.n_kv_heads);
  alloc_device(&s.qkv,    qkv_tot*sizeof(float), 0.f, true);
  alloc_device(&s.q,      c.head_dim*c.n_attn_heads*sizeof(float), 0.f, true);
  alloc_device(&s.att,    (c.n_attn_heads*(c.seq_len+1))*sizeof(float), 0.f, true);
  alloc_device(&s.logits, c.vocab_size*sizeof(float), 0.f, true);

  int kv_dim = c.head_dim * c.n_kv_heads;
  size_t cache_elems = 1ll*c.n_layers*c.seq_len*kv_dim;
  alloc_device(&s.key_cache,   cache_elems*sizeof(float), 0.f, true);
  alloc_device(&s.value_cache, cache_elems*sizeof(float), 0.f, true);

  if (c.sliding_window > 0) {
    alloc_device(&s.mask, 1ll*c.seq_len*c.seq_len*sizeof(float), 0.f, true);
    // initialize mask same as device 0 code: caller should copy host mask into this device mask if needed.
    float *hmask = (float*)malloc(1ll*c.seq_len*c.seq_len*sizeof(float));
    for (int i=0;i<c.seq_len;i++) for (int j=0;j<c.seq_len;j++) {
      float v = 0.f;
      if (c.sliding_window > 0 && i - j >= c.sliding_window) v = -INFINITY;
      hmask[i*c.seq_len + j] = v;
    }
    HIP_CHECK(hipMemcpy(s.mask, hmask, 1ll*c.seq_len*c.seq_len*sizeof(float), hipMemcpyHostToDevice));
    free(hmask);
  } else {
    s.mask = nullptr;
  }
}

// Helper: free RunState fields (call with hipSetDevice(dev) to free device-side memory)
static void free_runstate_on_device(RunState &s) {
  auto F = [&](float *&p){ if (p){ hipFree(p); p=nullptr; } };
  if (s.topk_v) hipFree(s.topk_v);
  if (s.topk_i) hipFree(s.topk_i);
  F(s.x); F(s.t); F(s.tb); F(s.tb2); F(s.router_score); F(s.mlp1_out);
  F(s.gate); F(s.up); F(s.gate_up); F(s.e_agg);
  F(s.qkv); F(s.q); F(s.att); F(s.logits); F(s.key_cache); F(s.value_cache);
  if (s.mask) { hipFree(s.mask); s.mask = nullptr; }
}

static void alloc_batchstate_on_device(BatchState &bs, const Config &c) {
  // ====== Host-side per-request arrays ======
  bs.positions         = (int  *)calloc((size_t)MAX_BATCH_SIZE, sizeof(int));
  bs.num_prompt_tokens = (int  *)calloc((size_t)MAX_BATCH_SIZE, sizeof(int));
  bs.prompt_tokens     = (int **)calloc((size_t)MAX_BATCH_SIZE, sizeof(int*)); // các phần tử sẽ malloc/free trong inference
  bs.current_tokens    = (int  *)calloc((size_t)MAX_BATCH_SIZE, sizeof(int));
  bs.finished          = (bool *)calloc((size_t)MAX_BATCH_SIZE, sizeof(bool));
  bs.req_ids           = (int  *)calloc((size_t)MAX_BATCH_SIZE, sizeof(int));

  // logits: host mảng con trỏ & device buffer chứa logits liên tiếp
  bs.logits_batch = (float**)malloc((size_t)MAX_BATCH_SIZE * sizeof(float*));

  // ====== Device-side batched buffers ======
  const int H   = c.hidden_dim;
  const int D   = c.head_dim;
  const int Hq  = c.n_attn_heads;
  const int Hkv = c.n_kv_heads;
  const int kv_dim = D * Hkv;
  const int qkv_tot = D * (Hq + 2 * Hkv);
  const size_t B = (size_t)MAX_BATCH_SIZE;

  // Embedding/residual stream
  alloc_device(&bs.batch_x,   B * (size_t)H * sizeof(float), 0.f, true);
  alloc_device(&bs.batch_t,   B * (size_t)H * sizeof(float), 0.f, true);

  // Attention projections
  alloc_device(&bs.batch_qkv, B * (size_t)qkv_tot * sizeof(float), 0.f, true);
  alloc_device(&bs.batch_q,   B * (size_t)(D * Hq) * sizeof(float), 0.f, true);

  // Attention caches: [n_layers, MAX_BATCH_SIZE, seq_len, kv_dim]
  {
    const size_t cache_elems = (size_t)c.n_layers * B * (size_t)c.seq_len * (size_t)kv_dim;
    alloc_device(&bs.batch_k, cache_elems * sizeof(float), 0.f, true);
    alloc_device(&bs.batch_v, cache_elems * sizeof(float), 0.f, true);
  }

  // Attention scores & outputs
  alloc_device(&bs.batch_att, B * (size_t)Hq * (size_t)(c.seq_len + 1) * sizeof(float), 0.f, true); // +1 cho sink
  alloc_device(&bs.batch_tb,  B * (size_t)(D * Hq) * sizeof(float), 0.f, true);
  alloc_device(&bs.batch_tb2, B * (size_t)H * sizeof(float), 0.f, true);

  // MLP / MoE
  alloc_device(&bs.batch_router_score, B * (size_t)c.n_experts * sizeof(float), 0.f, true);

  HIP_CHECK(hipMalloc((void**)&bs.batch_topk_v, B * (size_t)c.experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc((void**)&bs.batch_topk_i, B * (size_t)c.experts_per_token * sizeof(int)));

  alloc_device(&bs.batch_e_agg,     B * (size_t)H * sizeof(float), 0.f, true);
  alloc_device(&bs.batch_mlp1_out,  B * (size_t)(2 * c.intermediate_dim) * sizeof(float), 0.f, true);
  alloc_device(&bs.batch_gate,      B * (size_t)c.intermediate_dim * sizeof(float), 0.f, true);
  alloc_device(&bs.batch_up,        B * (size_t)c.intermediate_dim * sizeof(float), 0.f, true);
  alloc_device(&bs.batch_gate_up,   B * (size_t)c.intermediate_dim * sizeof(float), 0.f, true);
  
  // Logits buffer (device) + map host con trỏ
  alloc_device(&bs.batch_logits, B * (size_t)c.vocab_size * sizeof(float), 0.f, true);
  for (int i = 0; i < MAX_BATCH_SIZE; ++i) {
    bs.logits_batch[i] = bs.batch_logits + (size_t)i * (size_t)c.vocab_size;
  }

  alloc_device(&bs.batch_wexps, B * sizeof(float), 0.f, true);
}

static void free_batchstate_on_device(BatchState &bs) {
  auto Ff = [&](float *&p){ if (p) { hipFree(p); p = nullptr; } };
  auto Fi = [&](int   *&p){ if (p) { hipFree(p); p = nullptr; } };

  // Device frees
  Ff(bs.batch_x);
  Ff(bs.batch_t);
  Ff(bs.batch_qkv);
  Ff(bs.batch_q);
  Ff(bs.batch_k);
  Ff(bs.batch_v);
  Ff(bs.batch_att);
  Ff(bs.batch_tb);
  Ff(bs.batch_tb2);
  Ff(bs.batch_router_score);
  if (bs.batch_topk_v) { hipFree(bs.batch_topk_v); bs.batch_topk_v = nullptr; }
  if (bs.batch_topk_i) { hipFree(bs.batch_topk_i); bs.batch_topk_i = nullptr; }
  Ff(bs.batch_e_agg);
  Ff(bs.batch_mlp1_out);
  Ff(bs.batch_gate);
  Ff(bs.batch_up);
  Ff(bs.batch_gate_up);
  Ff(bs.batch_logits);
  Ff(bs.batch_wexps);
  Ff(bs.mask);

  // Host frees
  if (bs.prompt_tokens) {
    // Nếu inference đã free từng prompt_tokens[i] và set NULL thì vòng này an toàn.
    for (int i = 0; i < MAX_BATCH_SIZE; ++i) {
      if (bs.prompt_tokens[i]) { free(bs.prompt_tokens[i]); bs.prompt_tokens[i] = NULL; }
    }
    free(bs.prompt_tokens); bs.prompt_tokens = NULL;
  }
  if (bs.positions)         { free(bs.positions);         bs.positions = NULL; }
  if (bs.num_prompt_tokens) { free(bs.num_prompt_tokens); bs.num_prompt_tokens = NULL; }
  if (bs.current_tokens)    { free(bs.current_tokens);    bs.current_tokens = NULL; }
  if (bs.finished)          { free(bs.finished);          bs.finished = NULL; }
  if (bs.logits_batch)      { free(bs.logits_batch);      bs.logits_batch = NULL; }
  if (bs.req_ids)           { free(bs.req_ids);           bs.req_ids = NULL; }
}

// Initialize MoE multi-GPU, scatter MoE weights from host-mapped checkpoint and allocate per-device state.
// Call this after load_checkpoint_gpu(...) and after malloc_state_gpu(T) so T->data (host mmap) exists.
static void init_moe_gpu(Transformer *T, int requested_ngpus = 0) {
  const Config &c = T->config;
  int available = 0;
  HIP_CHECK(hipGetDeviceCount(&available));
  // pick how many GPUs to use for expert parallelism
  int ng = (requested_ngpus > 0 && requested_ngpus <= available) ? requested_ngpus : available;
  if (ng <= 1) {
    MOE_NGPUS = 1;
    MOE_GROUP_SIZE = c.n_experts;
    return;
  }
  MOE_NGPUS = ng;
  MOE_GROUP_SIZE = c.n_experts / MOE_NGPUS;
  if (MOE_GROUP_SIZE * MOE_NGPUS != c.n_experts) {
    fprintf(stderr, "MOE partitioning requires n_experts divisible by ngpus\n");
    exit(1);
  }

  // allocate arrays
  MOE_dev_w_mlp1 = (float**)malloc(sizeof(float*) * MOE_NGPUS);
  MOE_dev_w_mlp2 = (float**)malloc(sizeof(float*) * MOE_NGPUS);
  MOE_dev_b_mlp1 = (float**)malloc(sizeof(float*) * MOE_NGPUS);
  MOE_dev_b_mlp2 = (float**)malloc(sizeof(float*) * MOE_NGPUS);
  MOE_partial_on_dev0 = (float**)malloc(sizeof(float*) * MOE_NGPUS);
  MOE_dev_state = (RunState*)malloc(sizeof(RunState) * MOE_NGPUS);
  memset(MOE_dev_state, 0, sizeof(RunState) * MOE_NGPUS);
  
  MOE_batch_partial_on_dev0 = (float**)malloc(sizeof(float*) * MOE_NGPUS);
  MOE_batch_dev_state = (BatchState*)malloc(sizeof(BatchState) * MOE_NGPUS);
  memset(MOE_batch_dev_state, 0, sizeof(BatchState) * MOE_NGPUS);

  // compute host base pointer for MoE weights inside the mmap'd file:
  float *host_base = nullptr;
  if (T->data == nullptr) {
    fprintf(stderr, "init_moe_gpu: T->data (host mmap) is null - cannot scatter weights\n");
    exit(1);
  }
  host_base = T->data + sizeof(Config)/sizeof(float);
  float *ptr = host_base;

  // Walk offsets in same order as memory_map_weights_gpu to reach w_mlp1,b_mlp1,w_mlp2,b_mlp2
  ptr += 1ll * c.vocab_size * c.hidden_dim; // token_embedding_table
  ptr += 1ll * c.vocab_size * c.hidden_dim; // out
  ptr += 1ll * c.n_layers * c.hidden_dim;   // rms_attn_w
  ptr += 1ll * c.n_layers * c.hidden_dim;   // rms_ffn_w
  ptr += 1ll * c.hidden_dim;                // rms_out_w
  ptr += 1ll * c.n_layers * c.hidden_dim * (c.head_dim * c.n_attn_heads + 2 * c.head_dim * c.n_kv_heads); // w_qkv
  ptr += 1ll * c.n_layers * (c.head_dim * c.n_attn_heads + 2 * c.head_dim * c.n_kv_heads); // b_qkv
  ptr += 1ll * c.n_layers * (c.head_dim * c.n_attn_heads) * c.hidden_dim; // w_o
  ptr += 1ll * c.n_layers * c.hidden_dim; // b_o
  ptr += 1ll * c.n_layers * c.n_attn_heads; // attn_sinks
  ptr += 1ll * c.n_layers * c.hidden_dim * c.n_experts; // w_router
  ptr += 1ll * c.n_layers * c.n_experts; // b_router

  // Now ptr points to start of w_mlp1
  float *host_w_mlp1 = ptr;
  size_t mlp1_per_expert = (size_t)2 * c.intermediate_dim * c.hidden_dim;
  ptr += 1ll * c.n_layers * c.n_experts * mlp1_per_expert;

  float *host_b_mlp1 = ptr;
  ptr += 1ll * c.n_layers * c.n_experts * (size_t)(2 * c.intermediate_dim);

  float *host_w_mlp2 = ptr;
  size_t mlp2_per_expert = (size_t)c.hidden_dim * c.intermediate_dim;
  ptr += 1ll * c.n_layers * c.n_experts * mlp2_per_expert;

  float *host_b_mlp2 = ptr;
  ptr += 1ll * c.n_layers * c.n_experts * (size_t)c.hidden_dim;

  // Now scatter per-device
  size_t dev_mlp1_elems_per_layer = (size_t)MOE_GROUP_SIZE * mlp1_per_expert;
  size_t dev_mlp2_elems_per_layer = (size_t)MOE_GROUP_SIZE * mlp2_per_expert;
  size_t dev_b1_elems_per_layer = (size_t)MOE_GROUP_SIZE * (2 * c.intermediate_dim);
  size_t dev_b2_elems_per_layer = (size_t)MOE_GROUP_SIZE * c.hidden_dim;

  for (int d=0; d<MOE_NGPUS; ++d) {
    HIP_CHECK(hipSetDevice(d));
    // allocate device-local shard buffers sized for all layers
    size_t dev_mlp1_total = (size_t)c.n_layers * dev_mlp1_elems_per_layer;
    size_t dev_b1_total   = (size_t)c.n_layers * dev_b1_elems_per_layer;
    size_t dev_mlp2_total = (size_t)c.n_layers * dev_mlp2_elems_per_layer;
    size_t dev_b2_total   = (size_t)c.n_layers * dev_b2_elems_per_layer;

    HIP_CHECK( hipMalloc((void**)&MOE_dev_w_mlp1[d], dev_mlp1_total * sizeof(float)) );
    HIP_CHECK( hipMalloc((void**)&MOE_dev_b_mlp1[d], dev_b1_total * sizeof(float)) );
    HIP_CHECK( hipMalloc((void**)&MOE_dev_w_mlp2[d], dev_mlp2_total * sizeof(float)) );
    HIP_CHECK( hipMalloc((void**)&MOE_dev_b_mlp2[d], dev_b2_total * sizeof(float)) );

    // copy per-layer slices from host into this device's contiguous buffer
    for (int l=0; l<c.n_layers; ++l) {
      float *h_slice_w1 = host_w_mlp1 + (size_t)l * c.n_experts * mlp1_per_expert
                               + (size_t)d * MOE_GROUP_SIZE * mlp1_per_expert;
      float *dst_w1 = MOE_dev_w_mlp1[d] + (size_t)l * dev_mlp1_elems_per_layer;
      HIP_CHECK( hipMemcpy(dst_w1, h_slice_w1, dev_mlp1_elems_per_layer * sizeof(float), hipMemcpyHostToDevice) );

      float *h_slice_b1 = host_b_mlp1 + (size_t)l * c.n_experts * (2 * c.intermediate_dim)
                               + (size_t)d * MOE_GROUP_SIZE * (2 * c.intermediate_dim);
      float *dst_b1 = MOE_dev_b_mlp1[d] + (size_t)l * dev_b1_elems_per_layer;
      HIP_CHECK( hipMemcpy(dst_b1, h_slice_b1, dev_b1_elems_per_layer * sizeof(float), hipMemcpyHostToDevice) );

      float *h_slice_w2 = host_w_mlp2 + (size_t)l * c.n_experts * mlp2_per_expert
                               + (size_t)d * MOE_GROUP_SIZE * mlp2_per_expert;
      float *dst_w2 = MOE_dev_w_mlp2[d] + (size_t)l * dev_mlp2_elems_per_layer;
      HIP_CHECK( hipMemcpy(dst_w2, h_slice_w2, dev_mlp2_elems_per_layer * sizeof(float), hipMemcpyHostToDevice) );

      float *h_slice_b2 = host_b_mlp2 + (size_t)l * c.n_experts * c.hidden_dim
                               + (size_t)d * MOE_GROUP_SIZE * c.hidden_dim;
      float *dst_b2 = MOE_dev_b_mlp2[d] + (size_t)l * dev_b2_elems_per_layer;
      HIP_CHECK( hipMemcpy(dst_b2, h_slice_b2, dev_b2_elems_per_layer * sizeof(float), hipMemcpyHostToDevice) );
    }

    // allocate partial result buffer on device 0 (host-visible) for gathering; allocate on device 0
    HIP_CHECK(hipSetDevice(0));
    // HIP_CHECK( hipMalloc((void**)&MOE_partial_on_dev0[d], c.hidden_dim * sizeof(float)) );
    HIP_CHECK( hipMalloc((void**)&MOE_batch_partial_on_dev0[d], MAX_BATCH_SIZE * c.hidden_dim * sizeof(float)) );

    // allocate per-device RunState
    HIP_CHECK(hipSetDevice(d));
    // alloc_runstate_on_device(MOE_dev_state[d], c);
    alloc_batchstate_on_device(MOE_batch_dev_state[d], c);
  }

  // note: we keep device0's full T->weights*(all) untouched; the MoE shards additionally exist on each device.
  // Optionally we could free the full-device copy of MoE weights on device0 to save memory if necessary.
}

// cleanup
static void free_moe_gpu(Transformer *T) {
  const Config &c = T->config;
  if (MOE_NGPUS <= 1) return;
  for (int d=0; d<MOE_NGPUS; ++d) {
    // free per-device shards
    HIP_CHECK( hipSetDevice(d) );
    if (MOE_dev_w_mlp1 && MOE_dev_w_mlp1[d]) hipFree(MOE_dev_w_mlp1[d]);
    if (MOE_dev_w_mlp2 && MOE_dev_w_mlp2[d]) hipFree(MOE_dev_w_mlp2[d]);
    if (MOE_dev_b_mlp1 && MOE_dev_b_mlp1[d]) hipFree(MOE_dev_b_mlp1[d]);
    if (MOE_dev_b_mlp2 && MOE_dev_b_mlp2[d]) hipFree(MOE_dev_b_mlp2[d]);
    // free per-device RunState
    free_runstate_on_device(MOE_dev_state[d]);
    free_batchstate_on_device(MOE_batch_dev_state[d]);
    // free partial buffers on device 0
    HIP_CHECK( hipSetDevice(0) );
    if (MOE_partial_on_dev0 && MOE_partial_on_dev0[d]) hipFree(MOE_partial_on_dev0[d]);
    if (MOE_batch_partial_on_dev0 && MOE_batch_partial_on_dev0[d]) hipFree(MOE_batch_partial_on_dev0[d]);
  }
  if (MOE_dev_w_mlp1) free(MOE_dev_w_mlp1);
  if (MOE_dev_w_mlp2) free(MOE_dev_w_mlp2);
  if (MOE_dev_b_mlp1) free(MOE_dev_b_mlp1);
  if (MOE_dev_b_mlp2) free(MOE_dev_b_mlp2);
  if (MOE_partial_on_dev0) free(MOE_partial_on_dev0);
  if (MOE_dev_state) free(MOE_dev_state);

  MOE_dev_w_mlp1 = MOE_dev_w_mlp2 = MOE_dev_b_mlp1 = MOE_dev_b_mlp2 = nullptr;
  MOE_partial_on_dev0 = nullptr;
  MOE_dev_state = nullptr;
  MOE_NGPUS = 0; MOE_GROUP_SIZE = 0;

  if (MOE_batch_partial_on_dev0) free(MOE_batch_partial_on_dev0);
  if (MOE_batch_dev_state) free(MOE_batch_dev_state);
  MOE_batch_partial_on_dev0 = nullptr;
  MOE_batch_dev_state = nullptr;
}

// ------------------------------ I/O helpers ------------------------------

static void free_transformer_gpu(Transformer *T) {
  if (T->data && T->data!=MAP_FAILED) munmap(T->data, T->file_size);
  if (T->fd!=-1) close(T->fd);

  // free device weights/state
  free_moe_gpu(T); // free MoE multi-GPU infra if any
  TransformerWeights &g = T->weights;
  auto F=[&](float *&p){ if(p){ hipFree(p); p=nullptr; } };
  F(g.token_embedding_table); F(g.rms_attn_w); F(g.rms_ffn_w); F(g.w_qkv); F(g.w_o);
  F(g.b_qkv); F(g.b_o); F(g.attn_sinks); F(g.w_router); F(g.b_router);
  F(g.w_mlp1); F(g.w_mlp2); F(g.b_mlp1); F(g.b_mlp2); F(g.rms_out_w); F(g.out);

  // RunState &s = T->state;
  // F(s.x); F(s.t); F(s.tb); F(s.tb2); F(s.router_score); if(s.topk_v) hipFree(s.topk_v);
  // if(s.topk_i) hipFree(s.topk_i); F(s.mlp1_out); F(s.gate); F(s.up); F(s.gate_up); F(s.e_agg);
  // F(s.qkv); F(s.q); F(s.att); F(s.logits); F(s.key_cache); F(s.value_cache);
  // if (s.mask) hipFree(s.mask);
}

static void build_transformer_gpu(Transformer *T, char *ckpt) {
  T->fd = -1; T->data = nullptr; T->file_size = 0;
  // hipSetDevice(0); // MI250 GCD0
  load_checkpoint_gpu(ckpt, &T->config, &T->weights, &T->fd, &T->data, &T->file_size);
  // malloc_state_gpu(T);
  init_moe_gpu(T, 4); // use 4 GPUs for MoE expert parallelism
}

void malloc_batch_state_gpu(BatchState &bs, const Config &c) {
  alloc_batchstate_on_device(bs, c);

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

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the warm-up process
  // TODO:
  // - Memory allocation
  // - Load model
  // - ...
  char *checkpoint_path = "/nfs/gpu_trainee/final-project/modelbin/gpt-oss-20b.bin"; // e.g. out/model.bin
  const char *tokenizer_path = "tokenizer.bin";

  build_transformer_gpu(transformer, checkpoint_path);
  // read_tokenizer(tokenizer, tokenizer_path, transformer->config.vocab_size);

  HIP_CHECK( hipSetDevice(0) );
  g_batch_state = (BatchState*)calloc(1, sizeof(BatchState));
  malloc_batch_state_gpu(*g_batch_state, transformer->config);

  h_topk_i = (int*)  malloc((size_t)MAX_BATCH_SIZE * (size_t)transformer->config.experts_per_token * sizeof(int));
  h_topk_v = (float*)malloc((size_t)MAX_BATCH_SIZE * (size_t)transformer->config.experts_per_token * sizeof(float));
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the finish process
  // TODO:
  // - Memory deallocation
  // - Unload model
  // - ...
  free_transformer_gpu(transformer);
  // free_tokenizer(tokenizer);

  HIP_CHECK( hipSetDevice(0) );
  free_batchstate_on_device(*g_batch_state);
  free(g_batch_state); 
  g_batch_state = NULL;
  
  if (h_topk_i) {
    free(h_topk_i);
    h_topk_i = nullptr;
  }
  if (h_topk_v) {
    free(h_topk_v);
    h_topk_v = nullptr;
  }
}

long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, const char *input_seq,
                               int *output_tokens, int steps) {
  // <|start|>: 200006
  // <|end|>: 200007
  // <|return|>: 200002
  // <|message|>: 200008
  // <|channel|>: 200005
  // <|constrain|>: 200003
  // <|endoftext|>: 199999

  // Inference here

  const char *empty_prompt = "";
  if (input_seq == NULL) {
    input_seq = empty_prompt;
  }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(input_seq) + 3) *
                                     sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, input_seq, -1, -1, prompt_tokens, &num_prompt_tokens,
         transformer->config.initial_context_length);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // start the main loop
  int next;                     // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;                  // position in the sequence

  // print the very first token
  // should be removed
  const char *first_piece = decode_piece(tokenizer, 200006, token);
  safe_printf(first_piece);
  fflush(stdout);

  while (pos < steps) {

    // forward the transformer to get logits for the next token
    hipSetDevice(0);
    float *logits = forward_gpu(transformer, token, pos);

    // advance the state machine
    pos++;
    if (pos < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt
      // token
      next = prompt_tokens[pos];
    } else {
      // otherwise sample the next token from the logits
      next = sample_gpu(sampler, logits);
      // save the output token, it will be printed to file
      output_tokens[pos - num_prompt_tokens] = next;
    }

    // data-dependent terminating condition: the EOS (=199999 or =200002) token
    // delimits sequences
    if (next == 199999 || next == 200002) {
      break;
    }

    // print the token as string, decode it with the Tokenizer object
    // should be removed
    // const char *piece = decode_piece(tokenizer, token, next);
    // safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    // fflush(stdout);

    token = next;
  }

  // should be removed
  // printf("\n");

  // Marker for end of sequence
  output_tokens[pos - num_prompt_tokens + 1] = -1;

  free(prompt_tokens);

  return pos - num_prompt_tokens + 1;
}

void forward_batch(Transformer *transformer, int batch_size) {
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;
  
  int hidden_dim = p->hidden_dim;
  int head_dim = p->head_dim;
  int kv_dim = p->head_dim * p->n_kv_heads;
  int kv_mul = p->n_attn_heads / p->n_kv_heads;
  int n_qkv_heads = p->n_attn_heads + 2 * p->n_kv_heads;
  const int row_stride = p->seq_len + 1;

  int* d_positions = nullptr;
  HIP_CHECK(hipMalloc(&d_positions, batch_size * sizeof(int)));
  HIP_CHECK(hipMemcpyAsync(d_positions, g_batch_state->positions,
                          batch_size * sizeof(int), hipMemcpyHostToDevice, 0));
  
  const int half = head_dim / 2;
  float *cosB = nullptr, *sinB = nullptr;
  HIP_CHECK(hipMalloc(&cosB, (size_t)batch_size * (size_t)half * sizeof(float)));
  HIP_CHECK(hipMalloc(&sinB, (size_t)batch_size * (size_t)half * sizeof(float)));

  #pragma omp parallel for num_threads(batch_size)
  for (int b = 0; b < batch_size; ++b) {
    if (g_batch_state->finished[b]) {
      continue;
    }
    float *x = g_batch_state->batch_x + b * hidden_dim;
    float *content_row = w->token_embedding_table + g_batch_state->current_tokens[b] * hidden_dim;
    HIP_CHECK(hipMemcpy(x, content_row, hidden_dim * sizeof(float), hipMemcpyDeviceToDevice));
  }

  float ntk_beta  = 32.0f;
  float ntk_alpha = 1.0f;
  rope_ensure_invfreq(head_dim,
                      p->rope_theta,
                      p->rope_scaling_factor,
                      p->initial_context_length,
                      ntk_beta, ntk_alpha, 0
                    );

  for (int l = 0; l < p->n_layers; ++l) {
    rmsnorm_batch_gpu(
      g_batch_state->batch_t,
      g_batch_state->batch_x, 
      w->rms_attn_w + l * hidden_dim, 
      batch_size, 
      hidden_dim
    );

    gemv_gpu_batch(
      g_batch_state->batch_qkv, 
      g_batch_state->batch_t,  
      w->w_qkv + l * hidden_dim * (head_dim * n_qkv_heads),
      hidden_dim, 
      head_dim * n_qkv_heads, 
      batch_size
    );

    add_bias_gpu_batch_broadcast(
      g_batch_state->batch_qkv,
      w->b_qkv + l * head_dim * n_qkv_heads,
      batch_size,
      head_dim * n_qkv_heads
    );

    split_qkv_gpu_batch_hostidx(
      g_batch_state->batch_qkv,
      g_batch_state->batch_q,
      g_batch_state->batch_k,
      g_batch_state->batch_v,
      g_batch_state->positions,
      head_dim,
      p->n_attn_heads,
      p->n_kv_heads,
      batch_size,
      p->seq_len,
      l,
      MAX_BATCH_SIZE
    );

    rope_build_cos_sin_batch(
      cosB, sinB, 
      d_positions, 
      head_dim, batch_size, 0
    );

    rope_apply_q_batch(
      g_batch_state->batch_q,
      cosB, sinB,
      batch_size,
      p->n_attn_heads,
      head_dim, 0
    );

    rope_apply_k_batch(
      g_batch_state->batch_k,
      cosB, sinB, d_positions,
      batch_size,
      p->n_kv_heads,
      head_dim,
      p->seq_len,
      head_dim * p->n_kv_heads,
      l,
      MAX_BATCH_SIZE,
      0
    );

    const bool use_sw = (p->sliding_window > 0) && ((l % 2) == 0);

    attn_scores_gpu_batch(
      g_batch_state->batch_q,
      g_batch_state->batch_k + (size_t)l * (size_t)MAX_BATCH_SIZE * (size_t)p->seq_len * (size_t)kv_dim,
      use_sw ? g_batch_state->mask : nullptr,
      g_batch_state->batch_att,
      d_positions,
      head_dim, kv_mul, p->seq_len, kv_dim, p->n_attn_heads, batch_size,
      use_sw ? p->sliding_window : 0, 0);

    const float* sink_ptr = w->attn_sinks + (size_t)l * (size_t)p->n_attn_heads;
    softmax_rows_with_sink_gpu_batch(
      g_batch_state->batch_att, sink_ptr, d_positions,
      p->n_attn_heads, batch_size, row_stride, 0
    );

    attn_weighted_sum_gpu_batch(
      g_batch_state->batch_att,
      g_batch_state->batch_v + (size_t)l * (size_t)MAX_BATCH_SIZE * (size_t)p->seq_len * (size_t)kv_dim,
      g_batch_state->batch_tb,
      d_positions, head_dim, kv_mul, p->seq_len, kv_dim, p->n_attn_heads, batch_size, 0
    );
    
    const float *Wo = w->w_o + l * (head_dim * p->n_attn_heads) * hidden_dim;
    const float *Bo = w->b_o + l * hidden_dim;

    gemv_gpu_batch(
      g_batch_state->batch_tb2,
      g_batch_state->batch_tb,
      Wo,
      head_dim * p->n_attn_heads, 
      hidden_dim, 
      batch_size
    );

    add_bias_gpu_batch_broadcast(
      g_batch_state->batch_tb2,
      Bo,
      batch_size,
      hidden_dim
    );

    axpy_gpu_batch(
      g_batch_state->batch_x,
      g_batch_state->batch_tb2,
      1.0f,
      hidden_dim,
      batch_size
    );
    
    rmsnorm_batch_gpu(
      g_batch_state->batch_t,
      g_batch_state->batch_x,
      w->rms_ffn_w + l * hidden_dim,
      batch_size, hidden_dim
    );

    gemv_gpu_batch(
      g_batch_state->batch_router_score,
      g_batch_state->batch_t,
      w->w_router + l * hidden_dim * p->n_experts,
      hidden_dim,
      p->n_experts,
      batch_size
    );

    add_bias_gpu_batch_broadcast(
      g_batch_state->batch_router_score,
      w->b_router + l * p->n_experts,
      batch_size,
      p->n_experts
    );

    {

      rmsnorm_batch_gpu(
        g_batch_state->batch_t,
        g_batch_state->batch_x,
        w->rms_ffn_w + l * hidden_dim,
        batch_size, hidden_dim
      );

      gemv_gpu_batch(
        g_batch_state->batch_router_score,
        g_batch_state->batch_t,
        w->w_router + l * hidden_dim * p->n_experts,
        hidden_dim,
        p->n_experts,
        batch_size
      );

      add_bias_gpu_batch_broadcast(
        g_batch_state->batch_router_score,
        w->b_router + l * p->n_experts,
        batch_size,
        p->n_experts
      );

      topk_gpu_batch(
        g_batch_state->batch_topk_v,
        g_batch_state->batch_topk_i,
        g_batch_state->batch_router_score,
        batch_size,
        p->n_experts,
        p->experts_per_token,
        0
      );

      softmax_rows_gpu_batch_constlen(
        g_batch_state->batch_topk_v,
        batch_size,
        p->experts_per_token,
        p->experts_per_token,
        0
      );

      {
        const int BS = 256, GS = (batch_size * hidden_dim + BS - 1) / BS;
        hipLaunchKernelGGL(k_set, dim3(GS), dim3(BS), 0, 0, g_batch_state->batch_e_agg, 0.f, batch_size * hidden_dim);
      }

      if (MOE_NGPUS <= 1) {

      } else {
        const int H = hidden_dim;
        const int I = p->intermediate_dim;
        const int K = p->experts_per_token;

        // Lấy topk về host 1 lần (để quyết định owner)
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipMemcpy(h_topk_i, g_batch_state->batch_topk_i,
                            (size_t)batch_size * (size_t)K * sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_topk_v, g_batch_state->batch_topk_v,
                            (size_t)batch_size * (size_t)K * sizeof(float), hipMemcpyDeviceToHost));

        const size_t mlp1_per = (size_t)2 * (size_t)I * (size_t)H;
        const size_t mlp2_per = (size_t)H * (size_t)I;
        const size_t dev_mlp1_layer_stride = (size_t)MOE_GROUP_SIZE * mlp1_per;
        const size_t dev_mlp2_layer_stride = (size_t)MOE_GROUP_SIZE * mlp2_per;
        const size_t dev_b1_layer_stride   = (size_t)MOE_GROUP_SIZE * (size_t)(2 * I);
        const size_t dev_b2_layer_stride   = (size_t)MOE_GROUP_SIZE * (size_t)H;

        #pragma omp parallel for num_threads(MOE_NGPUS)
        for (int d = 0; d < MOE_NGPUS; ++d) {
          HIP_CHECK(hipSetDevice(d));
          BatchState &ds = MOE_batch_dev_state[d];

          const int BS = 256, GS = (batch_size * H + BS - 1) / BS;
          hipLaunchKernelGGL(k_set, dim3(GS), dim3(BS), 0, 0, ds.batch_e_agg, 0.f, batch_size * H);

          for (int e = 0; e < MOE_GROUP_SIZE; ++e) {
            int expert_id = d * MOE_GROUP_SIZE + e;

            int B = 0;

            for (int b = 0; b < batch_size; ++b) {
              for (int k = 0; k < K; ++k) {
                int top_e = h_topk_i[b * K + k];
                if (top_e == expert_id) {
                  ds.batch_wexps[B] = h_topk_v[b * K + k];
                  HIP_CHECK(hipMemcpyPeer(ds.batch_t + B * H, d,
                                          g_batch_state->batch_t + (size_t)b * (size_t)H, 0,
                                          (size_t)H * sizeof(float)));
                  ++B;
                  break;
                }
              }
            }
            
            if (B > 0) {
              const float *W1_local = MOE_dev_w_mlp1[d] + (size_t)l * dev_mlp1_layer_stride + (size_t)e * mlp1_per;
              const float *B1_local = MOE_dev_b_mlp1[d] + (size_t)l * dev_b1_layer_stride + (size_t)e * (size_t)(2 * I);

              gemv_gpu_batch(ds.batch_mlp1_out, ds.batch_t, W1_local, H, 2 * I, B);
              add_bias_gpu_batch_broadcast(ds.batch_mlp1_out, B1_local, B, 2 * I);

              {
                const int BS = 256, GS = (I * B + BS - 1) / BS;
                hipLaunchKernelGGL(split_gate_up, dim3(GS), dim3(BS), 0, 0,
                                  ds.batch_mlp1_out, ds.batch_gate, ds.batch_up, I * B);
              }

              swiglu_gpu_batch(ds.batch_gate, ds.batch_up, ds.batch_gate_up, I, 1.702f, p->swiglu_limit, B);

              const float *W2_local = MOE_dev_w_mlp2[d] + (size_t)l * dev_mlp2_layer_stride + (size_t)e * mlp2_per;
              const float *B2_local = MOE_dev_b_mlp2[d] + (size_t)l * dev_b2_layer_stride + (size_t)e * (size_t)H;

              gemv_gpu_batch(ds.batch_tb2, ds.batch_gate_up, W2_local, I, H, B);
              add_bias_gpu_batch_broadcast(ds.batch_tb2, B2_local, B, H);

              B = 0;
              for (int b = 0; b < batch_size; ++b) {
                for (int k = 0; k < K; ++k) {
                  int top_e = h_topk_i[b * K + k];
                  if (top_e == expert_id) {
                    axpy_gpu(ds.batch_e_agg + (size_t)b * (size_t)H,
                              ds.batch_tb2 + (size_t)B * (size_t)H,
                              ds.batch_wexps[B], H);
                    ++B;
                    break;
                  }
                }
              }

              // axpy_gpu_batch_alpha_vec(ds.batch_e_agg, ds.batch_tb2, ds.batch_wexps, H, B);
            }
          }

          HIP_CHECK(hipMemcpyPeer(MOE_batch_partial_on_dev0[d], 0,
                                  ds.batch_e_agg, d,
                                  (size_t)batch_size * (size_t)H * sizeof(float)));
        }

        // Reduce trên device 0: batch_e_agg[b] = sum_d partial[d][b]
        HIP_CHECK(hipSetDevice(0));
        for (int d = 0; d < MOE_NGPUS; ++d) {
          axpy_gpu_batch(
            g_batch_state->batch_e_agg,
            MOE_batch_partial_on_dev0[d],
            1.0f,
            H,
            batch_size
          );
        }
      }

      axpy_gpu_batch(
        g_batch_state->batch_x,
        g_batch_state->batch_e_agg,
        1.0f,
        hidden_dim,
        batch_size
      );
    }
  }

  rmsnorm_batch_gpu(
    g_batch_state->batch_x,
    g_batch_state->batch_x,
    w->rms_out_w,
    batch_size,
    hidden_dim
  );

  gemv_gpu_batch(
    g_batch_state->batch_logits,
    g_batch_state->batch_x,
    w->out,
    hidden_dim,
    p->vocab_size,
    batch_size
  );

  HIP_CHECK(hipFree(cosB));
  HIP_CHECK(hipFree(sinB));
  HIP_CHECK(hipFree(d_positions));

  HIP_CHECK(hipDeviceSynchronize());
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  HIP_CHECK(hipSetDevice(0));
  long long num_token_out = 0;
  
  const int num_reqs = requests->num_reqs;
  const int max_steps = requests->max_seq_len;
  const int batch_size = (num_reqs < MAX_BATCH_SIZE) ? num_reqs : MAX_BATCH_SIZE;
  const int vocab_size = transformer->config.vocab_size;

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

      const char *piece = decode_piece(tokenizer, g_batch_state->current_tokens[i], next_token);
      safe_printf(piece);
      fflush(stdout);

      g_batch_state->current_tokens[i] = next_token;

      if (next_token == 199999 || next_token == 200002 || pos >= max_steps) {
        g_batch_state->finished[i] = true;
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

#endif // GETP_RUN
