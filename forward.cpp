#include "batch_hip/rms_norm.cpp"
#include "batch_hip/matmul_bf16_X1D.cpp"
#include "batch_hip/matmul_bf16_X2D.cpp"
#include "batch_hip/matmul_fp32.cpp"
#include "batch_hip/add_bias.cpp"
#include "batch_hip/split_qkv.cpp"
#include "batch_hip/axpy.cpp"
#include "batch_hip/attention.cpp"
#include "batch_hip/swiglu.cpp"
#include "batch_hip/rope.cpp"
#include "batch_hip/softmax.cpp"
#include "batch_hip/top_k.cpp"
#include "batch_hip/split_gate_up.cpp"
#include "batch_hip/embedding_batch.cpp"
#include "batch_hip/moe.cpp"

static bool first_print = true;

void forward_batch(Transformer *transformer, int batch_size) {

   // ---------------- Profiling accumulators ----------------
  float t_embedding = 0, t_rmsnorm = 0, t_gemm_qkv = 0, t_bias_qkv = 0, t_split_qkv = 0;
  float t_rope = 0, t_attn_scores = 0, t_softmax_attn = 0, t_attn_weighted_sum = 0;
  float t_gemm_o = 0, t_bias_o = 0, t_axpy_tb2 = 0;
  float t_rmsnorm_ffn = 0, t_gemm_router = 0, t_bias_router = 0, t_topk = 0, t_softmax_moe = 0, t_topk_cpy = 0;
  float t_set_vec = 0,t_moe_gemm_mlp1 = 0, t_moe_bias_mlp1 = 0, t_moe_split_gate_up = 0, t_moe_swiglu = 0;
  float t_moe_gemm_mlp2 = 0, t_moe_bias_mlp2 = 0, t_wexps_cpy = 0, t_moe_axpy = 0, t_moe_axpy_agg = 0;
  float t_rmsnorm_out = 0, t_gemm_logits = 0;

  hipEvent_t ev_start, ev_stop;
  #define TIME_BLOCK(fn_call, t_accum) { \
    if (first_print) { \
      hipEventCreate(&ev_start); hipEventCreate(&ev_stop); \
      hipEventRecord(ev_start, 0); \
    } \
      fn_call; \
    if (first_print) { \
      hipEventRecord(ev_stop, 0); hipEventSynchronize(ev_stop); \
      float ms; hipEventElapsedTime(&ms, ev_start, ev_stop); \
      t_accum += ms; \
      hipEventDestroy(ev_start); hipEventDestroy(ev_stop); \
    } \
  }

  int device_id = 0;
  HIP_CHECK(hipGetDevice(&device_id));
  
  BatchState *g_batch_state = &batch_states[device_id];
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer_weights[device_id];
  
  int hidden_dim = p->hidden_dim;
  int head_dim = p->head_dim;
  int kv_dim = p->head_dim * p->n_kv_heads;
  int kv_mul = p->n_attn_heads / p->n_kv_heads;
  int n_qkv_heads = p->n_attn_heads + 2 * p->n_kv_heads;
  const int row_stride = p->seq_len + 1;

  int *d_current_tokens = g_batch_state->d_current_tokens;
  int *d_positions = g_batch_state->d_positions;
  float *cosB = g_batch_state->cosB;
  float *sinB = g_batch_state->sinB;
  
  hip_bfloat16 *d_w_mlp1_bf16 = g_batch_state->d_w_mlp1_bf16;
  hip_bfloat16 *d_w_mlp2_bf16 = g_batch_state->d_w_mlp2_bf16;

TIME_BLOCK({

  HIP_CHECK(hipMemcpyAsync(d_current_tokens, g_batch_state->current_tokens, 
                          batch_size * sizeof(int), hipMemcpyHostToDevice, 0));
                          
  if (hidden_dim % 4 == 0) {
    embedding_gather_vec4(g_batch_state->batch_x, w->token_embedding_table, d_current_tokens, batch_size, hidden_dim, 0);
  } else {
    embedding_gather(g_batch_state->batch_x, w->token_embedding_table, d_current_tokens, batch_size, hidden_dim, 0);
  }

  HIP_CHECK(hipMemcpyAsync(d_positions, g_batch_state->positions,
                          batch_size * sizeof(int), hipMemcpyHostToDevice, 0));

  float ntk_beta  = 32.0f;
  float ntk_alpha = 1.0f;
  rope_ensure_invfreq(head_dim,
                      p->rope_theta,
                      p->rope_scaling_factor,
                      p->initial_context_length,
                      ntk_beta, ntk_alpha, 0
                    );

}, t_embedding);

  for (int l = 0; l < p->n_layers; ++l) {
    
TIME_BLOCK({

    rmsnorm_batch_gpu(
      g_batch_state->batch_t,
      g_batch_state->batch_x, 
      w->rms_attn_w + 1ll * l * hidden_dim, 
      batch_size, 
      hidden_dim
    );

}, t_rmsnorm);

TIME_BLOCK({

    gemm_gpu_batch_f32W(
      g_batch_state->batch_qkv, 
      g_batch_state->batch_t,  
      w->w_qkv + 1ll * l * hidden_dim * (head_dim * n_qkv_heads),
      hidden_dim, 
      head_dim * n_qkv_heads, 
      batch_size
    );

}, t_gemm_qkv);

TIME_BLOCK({

    add_bias_gpu_batch_broadcast(
      g_batch_state->batch_qkv,
      w->b_qkv + 1ll * l * head_dim * n_qkv_heads,
      batch_size,
      head_dim * n_qkv_heads
    );

}, t_bias_qkv);

TIME_BLOCK({

    split_qkv_gpu_batch_devicepos(
      g_batch_state->batch_qkv,
      g_batch_state->batch_q,
      g_batch_state->batch_k,
      g_batch_state->batch_v,
      d_positions,
      head_dim,
      p->n_attn_heads,
      p->n_kv_heads,
      batch_size,
      p->seq_len,
      l,
      MAX_BATCH_SIZE
    );

}, t_split_qkv);

TIME_BLOCK({

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

}, t_rope);

    const bool use_sw = (p->sliding_window > 0) && ((l % 2) == 0);

TIME_BLOCK({

    attn_scores_gpu_batch(
      g_batch_state->batch_q,
      g_batch_state->batch_k + 1ll * l * MAX_BATCH_SIZE * p->seq_len * kv_dim,
      use_sw ? g_batch_state->mask : nullptr,
      g_batch_state->batch_att,
      d_positions,
      head_dim, kv_mul, p->seq_len, kv_dim, p->n_attn_heads, batch_size,
      use_sw ? p->sliding_window : 0, 0);

}, t_attn_scores);

TIME_BLOCK({

    const float* sink_ptr = w->attn_sinks + 1ll * l * p->n_attn_heads;
    softmax_rows_with_sink_gpu_batch(
      g_batch_state->batch_att, sink_ptr, d_positions,
      p->n_attn_heads, batch_size, row_stride, 0
    );

}, t_softmax_attn);

TIME_BLOCK({

    attn_weighted_sum_gpu_batch(
      g_batch_state->batch_att,
      g_batch_state->batch_v + 1ll * l * MAX_BATCH_SIZE * p->seq_len * kv_dim,
      g_batch_state->batch_tb,
      d_positions, head_dim, kv_mul, p->seq_len, kv_dim, p->n_attn_heads, batch_size, 0
    );

}, t_attn_weighted_sum);
    
    const float *Wo = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    const float *Bo = w->b_o + 1ll * l * hidden_dim;

TIME_BLOCK({

    gemm_gpu_batch_f32W(
      g_batch_state->batch_tb2,
      g_batch_state->batch_tb,
      Wo,
      head_dim * p->n_attn_heads, 
      hidden_dim, 
      batch_size
    );

}, t_gemm_o);

TIME_BLOCK({

    add_bias_gpu_batch_broadcast(
      g_batch_state->batch_tb2,
      Bo,
      batch_size,
      hidden_dim
    );

}, t_bias_o);

TIME_BLOCK({

    axpy_gpu_batch(
      g_batch_state->batch_x,
      g_batch_state->batch_tb2,
      1.0f,
      hidden_dim,
      batch_size
    );

}, t_axpy_tb2);
    
// ------------------------- MoE -------------------------

TIME_BLOCK({

    rmsnorm_batch_gpu(
      g_batch_state->batch_t,
      g_batch_state->batch_x,
      w->rms_ffn_w + l * hidden_dim,
      batch_size, hidden_dim
    );

}, t_rmsnorm_ffn);

TIME_BLOCK({

    gemm_gpu_batch_f32W(
      g_batch_state->batch_router_score,
      g_batch_state->batch_t,
      w->w_router + l * hidden_dim * p->n_experts,
      hidden_dim,
      p->n_experts,
      batch_size
    );

}, t_gemm_router);

TIME_BLOCK({

    add_bias_gpu_batch_broadcast(
      g_batch_state->batch_router_score,
      w->b_router + l * p->n_experts,
      batch_size,
      p->n_experts
    );

}, t_bias_router);

TIME_BLOCK({

    topk_gpu_batch(
      g_batch_state->batch_topk_v,
      g_batch_state->batch_topk_i,
      g_batch_state->batch_router_score,
      batch_size,
      p->n_experts,
      p->experts_per_token,
      0
    );

}, t_topk);

TIME_BLOCK({

    softmax_rows_gpu_batch_constlen(
      g_batch_state->batch_topk_v,
      batch_size,
      p->experts_per_token,
      p->experts_per_token,
      0
    );

}, t_softmax_moe);
    
    const int H = hidden_dim;
    const int I = p->intermediate_dim;
    const int K = p->experts_per_token;
    const int E = p->n_experts;

TIME_BLOCK({

    set(g_batch_state->d_counts, 0, E);

    moe_assign_from_topk(
      g_batch_state->batch_topk_i,
      g_batch_state->batch_topk_v,
      g_batch_state->batch_t,
      H, batch_size, K, E,
      g_batch_state->d_counts,
      g_batch_state->d_idx_in_batch,
      g_batch_state->d_in_ptrs,
      g_batch_state->d_wexps
    );

    
    HIP_CHECK(hipMemcpy(g_batch_state->h_counts, g_batch_state->d_counts, E * sizeof(int), hipMemcpyDeviceToHost));

}, t_set_vec);
    
    const long long mlp1_per = 2ll * I * H;
    const long long mlp2_per = 1ll * H * I;

    // #pragma omp parallel for
    for (int e = 0; e < E; e++) {
      int B = g_batch_state->h_counts[e];
      if (B > 0) {

        float **batch_t_gap = g_batch_state->d_in_ptrs + e * batch_size;
        float *batch_mlp1_out = g_batch_state->batch_mlp1_out;
        float *batch_gate = g_batch_state->batch_gate;
        float *batch_up = g_batch_state->batch_up;
        float *batch_gate_up = g_batch_state->batch_gate_up;
        float *d_out = g_batch_state->d_out + 1ll * e * batch_size * H;

        long long offset_l = 1ll * l * p->n_experts + e;

        const hip_bfloat16 *W1_local = d_w_mlp1_bf16 + offset_l * mlp1_per; 
        const float *B1_local = w->b_mlp1 + offset_l * (2 * I);

TIME_BLOCK({

        gemm_gpu_batch_bf16_2(batch_mlp1_out, batch_t_gap, W1_local, H, 2 * I, B);

}, t_moe_gemm_mlp1);

TIME_BLOCK({

        add_bias_gpu_batch_broadcast(batch_mlp1_out, B1_local, B, 2 * I);

}, t_moe_bias_mlp1);

TIME_BLOCK({

        split_gate_up(batch_mlp1_out, batch_gate, batch_up, I * B);

}, t_moe_split_gate_up);

TIME_BLOCK({

        swiglu_gpu_batch(batch_gate, batch_up, batch_gate_up, I, 1.702f, p->swiglu_limit, B);

}, t_moe_swiglu);

        const hip_bfloat16 *W2_local = d_w_mlp2_bf16 + offset_l * mlp2_per;
        const float *B2_local = w->b_mlp2 + offset_l * H;

TIME_BLOCK({

        gemm_gpu_batch_bf16(d_out, batch_gate_up, W2_local, I, H, B);

}, t_moe_gemm_mlp2);

TIME_BLOCK({

        add_bias_gpu_batch_broadcast(d_out, B2_local, B, H);

}, t_moe_bias_mlp2);

      }
    }

TIME_BLOCK({

    moe_aggregate_topk(
      g_batch_state->batch_topk_i,
      g_batch_state->batch_topk_v,
      g_batch_state->d_idx_in_batch,
      g_batch_state->d_out,
      g_batch_state->batch_x,
      H, batch_size, K, E
    );

}, t_moe_axpy_agg);
  
  }

TIME_BLOCK({

  rmsnorm_batch_gpu(
    g_batch_state->batch_x,
    g_batch_state->batch_x,
    w->rms_out_w,
    batch_size,
    hidden_dim
  );

}, t_rmsnorm_out);

TIME_BLOCK({

  gemm_gpu_batch_f32W(
    g_batch_state->batch_logits,
    g_batch_state->batch_x,
    w->out,
    hidden_dim,
    p->vocab_size,
    batch_size
  );

}, t_gemm_logits);

  if (!first_print) {
    return;
  }
  first_print = false;

  printf("===== Profiling (1 GPU, batched) =====\n");
  printf("[PROFILE] Embedding: %.3f ms\n", t_embedding);
  printf("[PROFILE] RMSNorm1: %.3f ms\n", t_rmsnorm);
  printf("[PROFILE] Gemm QKV: %.3f ms\n", t_gemm_qkv);
  printf("[PROFILE] Bias QKV: %.3f ms\n", t_bias_qkv);
  printf("[PROFILE] Split QKV: %.3f ms\n", t_split_qkv);
  printf("[PROFILE] RoPE: %.3f ms\n", t_rope);
  printf("[PROFILE] Attn scores: %.3f ms\n", t_attn_scores);
  printf("[PROFILE] Softmax attn: %.3f ms\n", t_softmax_attn);
  printf("[PROFILE] Attn weighted sum: %.3f ms\n", t_attn_weighted_sum);
  printf("[PROFILE] Gemm O: %.3f ms\n", t_gemm_o);
  printf("[PROFILE] Bias O: %.3f ms\n", t_bias_o);
  printf("[PROFILE] AXPY TB2: %.3f ms\n", t_axpy_tb2);
  printf("[PROFILE] RMSNorm2: %.3f ms\n", t_rmsnorm_ffn);
  printf("[PROFILE] Gemm router: %.3f ms\n", t_gemm_router);
  printf("[PROFILE] Bias router: %.3f ms\n", t_bias_router);
  printf("[PROFILE] TopK: %.3f ms\n", t_topk);
  printf("[PROFILE] Softmax MoE: %.3f ms\n", t_softmax_moe);
  printf("[PROFILE] TopK cpy: %.3f ms\n", t_topk_cpy);
  printf("[PROFILE] Set vec: %.3f ms\n", t_set_vec);
  printf("[PROFILE] MoE Gemm MLP1: %.3f ms\n", t_moe_gemm_mlp1);
  printf("[PROFILE] MoE Bias MLP1: %.3f ms\n", t_moe_bias_mlp1);
  printf("[PROFILE] MoE Split gate up: %.3f ms\n", t_moe_split_gate_up);
  printf("[PROFILE] MoE SwiGLU: %.3f ms\n", t_moe_swiglu);
  printf("[PROFILE] MoE Gemm MLP2: %.3f ms\n", t_moe_gemm_mlp2);
  printf("[PROFILE] MoE Bias MLP2: %.3f ms\n", t_moe_bias_mlp2);
  printf("[PROFILE] MoE Wexps cpy: %.3f ms\n", t_wexps_cpy);
  printf("[PROFILE] MoE AXPY: %.3f ms\n", t_moe_axpy);
  printf("[PROFILE] MoE Agg: %.3f ms\n", t_moe_axpy_agg);
  printf("[PROFILE] RMSNorm out: %.3f ms\n", t_rmsnorm_out);
  printf("[PROFILE] Gemm logits: %.3f ms\n", t_gemm_logits);

  float total = t_embedding + t_rmsnorm + t_gemm_qkv + t_bias_qkv + t_split_qkv +
                t_rope + t_attn_scores + t_softmax_attn + t_attn_weighted_sum +
                t_gemm_o + t_bias_o + t_axpy_tb2 +
                t_rmsnorm_ffn + t_gemm_router + t_bias_router + t_topk + t_softmax_moe + t_topk_cpy +
                t_set_vec + t_moe_gemm_mlp1 + t_moe_bias_mlp1 + t_moe_split_gate_up + t_moe_swiglu +
                t_moe_gemm_mlp2 + t_moe_bias_mlp2 + t_wexps_cpy + t_moe_axpy + t_moe_axpy_agg +
                t_rmsnorm_out + t_gemm_logits;
  printf("[PROFILE] Total: %.3f ms\n", total);
  printf("=======================================\n");
  fflush(stdout);
  // HIP_CHECK(hipDeviceSynchronize());
}
