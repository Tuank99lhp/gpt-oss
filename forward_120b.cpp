

inline void GemmQKV_120b(
    float* Y, const hip_bfloat16* X, const hip_bfloat16* W,
    int K, int M, int B, const float* bias = nullptr, hipStream_t s = 0)
{
  gemm_gpu_batch_bf16core_yfp32<256, 64, 128, 16, 1>(Y, X, W, K, M, B, bias, s);
}

inline void GemmO_120b(
    float* Y, const hip_bfloat16* X, const hip_bfloat16* W,
    int K, int M, int B, const float* bias = nullptr, hipStream_t s = 0)
{
  gemm_gpu_batch_bf16core_yfp32<256, 64, 256, 16, 1>(Y, X, W, K, M, B, bias, s);
}

inline void GemmRouter_120b(
    float* Y, const hip_bfloat16* X, const hip_bfloat16* W,
    int K, int M, int B, const float* bias = nullptr, hipStream_t s = 0)
{
  gemm_gpu_batch_bf16core_yfp32<128, 16, 512, 8, 1>(Y, X, W, K, M, B, bias, s);
}

inline void GemmMlp1_120b(
    float* Y, const hip_bfloat16* X, const hip_bfloat16* W,
    int K, int M, int B, const float* bias = nullptr, hipStream_t s = 0)
{
  gemm_gpu_batch_bf16core_yfp32<256, 64, 128, 16, 1>(Y, X, W, K, M, B, bias, s);
}

inline void GemmMlp2_120b(
    float* Y, const hip_bfloat16* X, const hip_bfloat16* W,
    int K, int M, int B, const float* bias = nullptr, hipStream_t s = 0)
{
  gemm_gpu_batch_bf16core_yfp32<256, 64, 128, 16, 1>(Y, X, W, K, M, B, bias, s);
}

inline void GemmLogits_120b(
    float* Y, const hip_bfloat16* X, const hip_bfloat16* W,
    int K, int M, int B, const float* bias = nullptr, hipStream_t s = 0)
{
  gemm_gpu_batch_bf16core_yfp32<256, 64, 128, 16, 1>(Y, X, W, K, M, B, bias, s);
}

void forward_batch_120b(Transformer *transformer, int batch_size) {

   // ---------------- Profiling accumulators ----------------
  float t_embedding = 0, t_rmsnorm = 0, t_gemm_qkv = 0, t_split_qkv = 0;
  float t_rope = 0, t_attn_scores = 0, t_softmax_attn = 0, t_attn_weighted_sum = 0;
  float t_gemm_o = 0, t_axpy_tb2 = 0;
  float t_rmsnorm_ffn = 0, t_gemm_router = 0, t_topk = 0, t_softmax_moe = 0;
  float t_set_vec = 0,t_moe_gemm_mlp1 = 0, t_moe_swiglu = 0;
  float t_moe_gemm_mlp2 = 0, t_moe_axpy_agg = 0;
  float t_split_moe = 0, t_agg_moe = 0;
  float t_rmsnorm_out = 0, t_gemm_logits = 0;

  hipEvent_t ev_start, ev_stop;

  int device_id = 0;
  HIP_CHECK(hipGetDevice(&device_id));
  assert(device_id == 0);
  
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

TIME_BLOCK({

  HIP_CHECK(hipMemcpyAsync(d_current_tokens, g_batch_state->current_tokens, 
                          batch_size * sizeof(int), hipMemcpyHostToDevice, 0));
                          
  if (hidden_dim % 4 == 0) {
    embedding_gather_vec4(g_batch_state->batch_x, g_batch_state->d_w_token_embedding_table_bf16, d_current_tokens, batch_size, hidden_dim, 0);
  } else {
    embedding_gather(g_batch_state->batch_x, g_batch_state->d_w_token_embedding_table_bf16, d_current_tokens, batch_size, hidden_dim, 0);
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

    rmsnorm_batch_gpu_bf16(
      g_batch_state->batch_t,
      g_batch_state->batch_x, 
      w->rms_attn_w + 1ll * l * hidden_dim, 
      batch_size, 
      hidden_dim
    );

}, t_rmsnorm);

TIME_BLOCK({

    GemmQKV_120b(
      g_batch_state->batch_qkv, 
      g_batch_state->batch_t,  
      g_batch_state->d_w_qkv_bf16 + 1ll * l * hidden_dim * (head_dim * n_qkv_heads),
      hidden_dim, 
      head_dim * n_qkv_heads, 
      batch_size,
      w->b_qkv + 1ll * l * head_dim * n_qkv_heads
    );

}, t_gemm_qkv);

TIME_BLOCK({

    split_qkv_gpu_batch_devicepos_f32q_bf16kv(
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
      MAX_BATCH_SIZE,
      p->sliding_window
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

    rope_apply_k_batch_bf16(
      g_batch_state->batch_k,
      cosB, sinB, d_positions,
      batch_size,
      p->n_kv_heads,
      head_dim,
      p->seq_len,
      head_dim * p->n_kv_heads,
      l,
      MAX_BATCH_SIZE,
      p->sliding_window
    );

}, t_rope);

    const bool use_sw = (p->sliding_window > 0) && ((l % 2) == 0);

TIME_BLOCK({

    attn_scores_gpu_batch_bf16k(
    g_batch_state->batch_q,
    g_batch_state->batch_k,                 // truyền base tổng, wrapper tự offset theo layer
    use_sw ? g_batch_state->mask : nullptr, // mask[pos, t] hoặc null
    g_batch_state->batch_att,
    d_positions,
    head_dim, kv_mul,
    p->seq_len, kv_dim,
    p->n_attn_heads, batch_size,
    /*layer=*/l,
    MAX_BATCH_SIZE,
    p->sliding_window,
    /*stream=*/0);

}, t_attn_scores);

TIME_BLOCK({

    const float* sink_ptr = w->attn_sinks + 1ll * l * p->n_attn_heads;
    softmax_rows_with_sink_gpu_batch(
      g_batch_state->batch_att, sink_ptr, d_positions,
      p->n_attn_heads, batch_size, row_stride, 0
    );

}, t_softmax_attn);

TIME_BLOCK({

    attn_weighted_sum_gpu_batch_bf16v(
    g_batch_state->batch_att,
    g_batch_state->batch_v,          // truyền base tổng; wrapper tự offset theo layer
    g_batch_state->batch_tb,
    d_positions,
    head_dim, kv_mul,
    p->seq_len, kv_dim,
    p->n_attn_heads, batch_size,
    /*layer=*/l,
    MAX_BATCH_SIZE,
    p->sliding_window,
    /*stream=*/0);

}, t_attn_weighted_sum);
    
    const hip_bfloat16 *Wo = g_batch_state->d_w_o_bf16 + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    const float *Bo = w->b_o + 1ll * l * hidden_dim;

TIME_BLOCK({

    GemmO_120b(
      g_batch_state->batch_tb2,
      g_batch_state->batch_tb,
      Wo,
      head_dim * p->n_attn_heads, 
      hidden_dim, 
      batch_size,
      Bo
    );

}, t_gemm_o);

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

    rmsnorm_batch_gpu_bf16(
      g_batch_state->batch_t,
      g_batch_state->batch_x,
      w->rms_ffn_w + l * hidden_dim,
      batch_size, hidden_dim
    );

}, t_rmsnorm_ffn);

TIME_BLOCK({

    GemmRouter_120b(
      g_batch_state->batch_router_score,
      g_batch_state->batch_t,
      g_batch_state->d_w_router_bf16 + l * hidden_dim * p->n_experts,
      hidden_dim,
      p->n_experts,
      batch_size,
      w->b_router + l * p->n_experts
    );

}, t_gemm_router);

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

    moe_assign_from_topk_bf16(
      g_batch_state->batch_topk_i,
      g_batch_state->batch_topk_v,
      g_batch_state->batch_t,
      H, batch_size, K, E,
      g_batch_state->d_counts,
      g_batch_state->d_idx_in_batch,
      g_batch_state->d_in_ptrs
    );

    HIP_CHECK(hipMemcpy(g_batch_state->h_counts, g_batch_state->d_counts, E * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(g_batch_state->h_in_ptrs, g_batch_state->d_in_ptrs, E * batch_size * sizeof(hip_bfloat16*), hipMemcpyDeviceToHost));

}, t_set_vec);
    
    const long long mlp1_per = 2ll * I * H;
    const long long mlp2_per = 1ll * H * I;

    #pragma omp parallel for
    for (int d = 1; d < NUM_GPUS; ++d) {
      HIP_CHECK(hipSetDevice(d));

      TransformerWeights *l_w = &transformer_weights[d];
      BatchState *l_batch_state = &batch_states[d];

      int div = E / (NUM_GPUS - 1);
      int mod = E % (NUM_GPUS - 1);
      int num_experts = div + (d <= mod);
      int st_idx = (d - 1) * div + min(d - 1, mod);
      
      for (int ep = st_idx; ep < st_idx + num_experts; ++ep) {
        int B = g_batch_state->h_counts[ep];
        if (B == 0) {
          continue;
        }

        for (int b = 0; b < B; ++b) {
          HIP_CHECK(hipMemcpyPeer(
            l_batch_state->batch_t + b * H, d,
            g_batch_state->h_in_ptrs[ep * batch_size + b], 0,
            H * sizeof(hip_bfloat16)));
        }

        float *batch_mlp1_out = l_batch_state->batch_mlp1_out;
        hip_bfloat16 *batch_gate_up = l_batch_state->batch_gate_up;
        float *d_out = g_batch_state->d_out + 1ll * ep * batch_size * H;

        long long offset_l = 1ll * l * num_experts + ep - st_idx;

        const hip_bfloat16 *W1_local = l_batch_state->d_w_mlp1_bf16 + offset_l * mlp1_per;
        const float *B1_local = l_w->b_mlp1 + offset_l * (2 * I);

        GemmMlp1_120b(batch_mlp1_out, l_batch_state->batch_t, W1_local, H, 2 * I, B, B1_local);

        swiglu_fused_gpu_batch_bf16(
          batch_mlp1_out,
          batch_gate_up,
          I, B,
          1.702f,
          p->swiglu_limit,
          0
        );

        const hip_bfloat16 *W2_local = l_batch_state->d_w_mlp2_bf16 + offset_l * mlp2_per;
        const float *B2_local = l_w->b_mlp2 + offset_l * H;

        GemmMlp2_120b(l_batch_state->batch_x, batch_gate_up, W2_local, I, H, B, B2_local);
  
        HIP_CHECK(hipMemcpyPeer(
          d_out, 0,
          l_batch_state->batch_x, d,
          B * H * sizeof(float)));

      }
    }

    assert(device_id == 0);
    HIP_CHECK(hipSetDevice(device_id));

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

  rmsnorm_batch_gpu_bf16(
    g_batch_state->batch_t,
    g_batch_state->batch_x,
    w->rms_out_w,
    batch_size,
    hidden_dim
  );

}, t_rmsnorm_out);

TIME_BLOCK({

  GemmLogits_120b(
    g_batch_state->batch_logits,
    g_batch_state->batch_t,
    g_batch_state->d_w_out_bf16,
    hidden_dim,
    p->vocab_size,
    batch_size
  );

}, t_gemm_logits);

  if (!first_print || device_id != 0) {
    return;
  }
  first_print = false;

  printf("===== Profiling (1 GPU, batched) =====\n");
  printf("[PROFILE] Embedding: %.3f ms\n", t_embedding);
  printf("[PROFILE] RMSNorm1: %.3f ms\n", t_rmsnorm);
  printf("[PROFILE] Gemm QKV: %.3f ms\n", t_gemm_qkv);
  printf("[PROFILE] Split QKV: %.3f ms\n", t_split_qkv);
  printf("[PROFILE] RoPE: %.3f ms\n", t_rope);
  printf("[PROFILE] Attn scores: %.3f ms\n", t_attn_scores);
  printf("[PROFILE] Softmax attn: %.3f ms\n", t_softmax_attn);
  printf("[PROFILE] Attn weighted sum: %.3f ms\n", t_attn_weighted_sum);
  printf("[PROFILE] Gemm O: %.3f ms\n", t_gemm_o);
  printf("[PROFILE] AXPY TB2: %.3f ms\n", t_axpy_tb2);
  printf("[PROFILE] RMSNorm2: %.3f ms\n", t_rmsnorm_ffn);
  printf("[PROFILE] Gemm router: %.3f ms\n", t_gemm_router);
  printf("[PROFILE] TopK: %.3f ms\n", t_topk);
  printf("[PROFILE] Softmax MoE: %.3f ms\n", t_softmax_moe);
  printf("[PROFILE] Set vec: %.3f ms\n", t_set_vec);
  printf("[PROFILE] MoE memcy batch_t: %.3f ms\n", t_split_moe);
  printf("[PROFILE] MoE Gemm MLP1: %.3f ms\n", t_moe_gemm_mlp1);
  printf("[PROFILE] MoE SwiGLU: %.3f ms\n", t_moe_swiglu);
  printf("[PROFILE] MoE Gemm MLP2: %.3f ms\n", t_moe_gemm_mlp2);
  printf("[PROFILE] MoE Agg: %.3f ms\n", t_agg_moe);
  printf("[PROFILE] MoE AXPY agg: %.3f ms\n", t_moe_axpy_agg);
  printf("[PROFILE] RMSNorm out: %.3f ms\n", t_rmsnorm_out);
  printf("[PROFILE] Gemm logits: %.3f ms\n", t_gemm_logits);

  float total = t_embedding + t_rmsnorm + t_gemm_qkv + t_split_qkv +
                t_rope + t_attn_scores + t_softmax_attn + t_attn_weighted_sum +
                t_gemm_o + t_axpy_tb2 +
                t_rmsnorm_ffn + t_gemm_router + t_topk + t_softmax_moe +
                t_set_vec + t_moe_gemm_mlp1 + t_moe_swiglu +
                t_moe_gemm_mlp2 + t_moe_axpy_agg +
                t_rmsnorm_out + t_gemm_logits + t_split_moe + t_agg_moe;
  printf("[PROFILE] Total: %.3f ms\n", total);
  printf("=======================================\n");
  fflush(stdout);
  // HIP_CHECK(hipDeviceSynchronize());
}
