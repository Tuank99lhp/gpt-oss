#include "batch_hip/rms_norm.cpp"
#include "batch_hip/matmul.cpp"
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

void forward_batch(Transformer *transformer, int batch_size) {
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

  int *h_topk_i = g_batch_state->h_topk_i;
  float *h_topk_v = g_batch_state->h_topk_v;

  int *d_current_tokens = g_batch_state->d_current_tokens;
  int *d_positions = g_batch_state->d_positions;
  float *cosB = g_batch_state->cosB;
  float *sinB = g_batch_state->sinB;
  
  hip_bfloat16 *d_w_mlp1_bf16 = g_batch_state->d_w_mlp1_bf16;
  hip_bfloat16 *d_w_mlp2_bf16 = g_batch_state->d_w_mlp2_bf16;

  BatchStateMOE* MOE_tmp_batch_state = g_batch_state->MOE_tmp_batch_state;

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

  for (int l = 0; l < p->n_layers; ++l) {
    rmsnorm_batch_gpu(
      g_batch_state->batch_t,
      g_batch_state->batch_x, 
      w->rms_attn_w + 1ll * l * hidden_dim, 
      batch_size, 
      hidden_dim
    );

    gemm_gpu_batch_f32W(
      g_batch_state->batch_qkv, 
      g_batch_state->batch_t,  
      w->w_qkv + 1ll * l * hidden_dim * (head_dim * n_qkv_heads),
      hidden_dim, 
      head_dim * n_qkv_heads, 
      batch_size
    );

    add_bias_gpu_batch_broadcast(
      g_batch_state->batch_qkv,
      w->b_qkv + 1ll * l * head_dim * n_qkv_heads,
      batch_size,
      head_dim * n_qkv_heads
    );

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
      g_batch_state->batch_k + 1ll * l * MAX_BATCH_SIZE * p->seq_len * kv_dim,
      use_sw ? g_batch_state->mask : nullptr,
      g_batch_state->batch_att,
      d_positions,
      head_dim, kv_mul, p->seq_len, kv_dim, p->n_attn_heads, batch_size,
      use_sw ? p->sliding_window : 0, 0);

    const float* sink_ptr = w->attn_sinks + 1ll * l * p->n_attn_heads;
    softmax_rows_with_sink_gpu_batch(
      g_batch_state->batch_att, sink_ptr, d_positions,
      p->n_attn_heads, batch_size, row_stride, 0
    );

    attn_weighted_sum_gpu_batch(
      g_batch_state->batch_att,
      g_batch_state->batch_v + 1ll * l * MAX_BATCH_SIZE * p->seq_len * kv_dim,
      g_batch_state->batch_tb,
      d_positions, head_dim, kv_mul, p->seq_len, kv_dim, p->n_attn_heads, batch_size, 0
    );
    
    const float *Wo = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    const float *Bo = w->b_o + 1ll * l * hidden_dim;

    gemm_gpu_batch_f32W(
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
    
    {

      rmsnorm_batch_gpu(
        g_batch_state->batch_t,
        g_batch_state->batch_x,
        w->rms_ffn_w + l * hidden_dim,
        batch_size, hidden_dim
      );

      gemm_gpu_batch_f32W(
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
      
      const int H = hidden_dim;
      const int I = p->intermediate_dim;
      const int K = p->experts_per_token;

      HIP_CHECK(hipMemcpy(h_topk_i, g_batch_state->batch_topk_i, batch_size * K * sizeof(int), hipMemcpyDeviceToHost)); 
      HIP_CHECK(hipMemcpy(h_topk_v, g_batch_state->batch_topk_v, batch_size * K * sizeof(float), hipMemcpyDeviceToHost));

      const long long mlp1_per = 2ll * I * H;
      const long long mlp2_per = 1ll * H * I;

      #pragma omp parallel for
      for (int e = 0; e < p->n_experts; e++) {
        BatchStateMOE &ds = MOE_tmp_batch_state[e];

        int &B = ds.num_batch; 
        B = 0;

        for (int b = 0; b < batch_size; ++b) {
          for (int k = 0; k < K; ++k) {
            if (h_topk_i[b * K + k] == e) {
              ds.host_wexps[B] = h_topk_v[b * K + k];
              set_vec(ds.batch_t + 1ll * B * H, g_batch_state->batch_t + 1ll * b * H, H);
              ds.idx_in_batch[B] = b;
              ++B;
              break;
            }
          }
        }

        if (B > 0) {
          long long offset_l = 1ll * l * p->n_experts + e;

          const hip_bfloat16 *W1_local = d_w_mlp1_bf16 + offset_l * mlp1_per; 
          const float *B1_local = w->b_mlp1 + offset_l * (2 * I);

          gemm_gpu_batch_bf16(ds.batch_mlp1_out, ds.batch_t, W1_local, H, 2 * I, B);
          add_bias_gpu_batch_broadcast(ds.batch_mlp1_out, B1_local, B, 2 * I);

          split_gate_up(ds.batch_mlp1_out, ds.batch_gate, ds.batch_up, I * B);

          swiglu_gpu_batch(ds.batch_gate, ds.batch_up, ds.batch_gate_up, I, 1.702f, p->swiglu_limit, B);

          const hip_bfloat16 *W2_local = d_w_mlp2_bf16 + offset_l * mlp2_per;
          const float *B2_local = w->b_mlp2 + offset_l * H;

          gemm_gpu_batch_bf16(ds.batch_t, ds.batch_gate_up, W2_local, I, H, B);
          add_bias_gpu_batch_broadcast(ds.batch_t, B2_local, B, H);

          HIP_CHECK(hipMemcpy(ds.batch_wexps, ds.host_wexps, B * sizeof(float), hipMemcpyHostToDevice));
          xpy_gpu_batch_alpha_vec(ds.batch_t, ds.batch_t, ds.batch_wexps, H, B);
        }
      }

      for (int e = 0; e < p->n_experts; ++e) {
        BatchStateMOE &ds = MOE_tmp_batch_state[e];
        int B = ds.num_batch;
        #pragma omp parallel for
        for (int b = 0; b < B; b++) {
          int ob = ds.idx_in_batch[b];
          axpy_gpu_batch(g_batch_state->batch_x + 1ll * ob * H,
                   ds.batch_t + 1ll * b * H,
                   1.0f, H, 1);
        }
      }
    }
  }

  rmsnorm_batch_gpu(
    g_batch_state->batch_x,
    g_batch_state->batch_x,
    w->rms_out_w,
    batch_size,
    hidden_dim
  );

  gemm_gpu_batch_f32W(
    g_batch_state->batch_logits,
    g_batch_state->batch_x,
    w->out,
    hidden_dim,
    p->vocab_size,
    batch_size
  );

  // HIP_CHECK(hipDeviceSynchronize());
}
