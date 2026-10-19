#include "check.h"
#include "ops.h"
#include "transformer.h"

static void test_rmsnorm_formula(void) {
  Matrix x = mat_create(1, 3), w = mat_create(1, 3), out = mat_create(1, 3);
  x.data[0] = 3;
  x.data[1] = -4;
  x.data[2] = 0;
  for (int i = 0; i < 3; i++)
    w.data[i] = 1.0f; // weights 1 = "change nothing", isolates the normalization

  rmsnorm(out, x, w);
  CHECK_NEAR(out.data[0], 1.03923f, 1e-5f);
  CHECK_NEAR(out.data[1], -1.38564f, 1e-5f);
  CHECK_NEAR(out.data[2], 0.0f, 1e-5f);

  mat_free(x);
  mat_free(w);
  mat_free(out);
}

static void test_silu(void) {
  Matrix z = mat_create(1, 3);
  z.data[0] = 2;
  z.data[1] = 0;
  z.data[2] = -2;
  silu(z);
  CHECK_NEAR(z.data[0], 1.761594f, 1e-5f);
  CHECK_NEAR(z.data[1], 0.0f, 1e-6f);
  CHECK_NEAR(z.data[2], -0.238406f, 1e-5f);
  mat_free(z);
}

// Expected values were read straight from the file with `od` (see notes).
int main(void) {
  test_rmsnorm_formula();
  test_silu();

  Transformer t;
  load_transformer(&t, "models/stories15M.bin"); // exits on end-of-file mismatch

  Config c = t.config;
  CHECK(c.dim == 288 && c.hidden_dim == 768 && c.n_layers == 6);
  CHECK(c.n_heads == 6 && c.n_kv_heads == 6 && c.vocab_size == 32000 && c.seq_len == 256);
  CHECK(t.shared_weights);

  CHECK(t.token_embedding.rows == 32000 && t.token_embedding.columns == 288);
  CHECK_NEAR(t.token_embedding.data[0], -0.0598246f, 1e-6f);
  CHECK_NEAR(t.rms_att[1].data[0], 1.1206539f, 1e-6f);
  CHECK_NEAR(t.wq[0].data[288], -0.0045094f, 1e-6f); // row 1
  CHECK_NEAR(t.wk[5].data[0], -0.0052079f, 1e-6f);
  CHECK_NEAR(t.rms_ffn[0].data[0], 1.0146376f, 1e-6f);
  CHECK(t.w1[0].rows == 768 && t.w1[0].columns == 288);
  CHECK(t.w2[0].rows == 288 && t.w2[0].columns == 768);
  CHECK_NEAR(t.w2[0].data[768], 0.0093553f, 1e-6f); // row 1: 768 floats in
  CHECK_NEAR(t.rms_final.data[0], 7.6768489f, 1e-6f);
  CHECK(t.wcls.data == t.token_embedding.data);

  RunState state = run_state_create(c);
  embed(&t, &state, 1); // BOS

  CHECK(state.x.rows == 1 && state.x.columns == 288);
  CHECK(state.x.owns_data);
  CHECK(state.x.data != t.token_embedding.data + 288); // a copy, not a view
  for (int i = 0; i < 288; i++)
    CHECK(state.x.data[i] == t.token_embedding.data[288 + i]);
  CHECK_NEAR(state.x.data[0], -0.0170964f, 1e-6f); // from od, byte 1180

  rmsnorm(state.xnorm, state.x, t.rms_att[0]);
  CHECK_NEAR(state.xnorm.data[0], -0.577490f, 1e-5f);
  CHECK_NEAR(state.xnorm.data[1], -0.456318f, 1e-5f);
  CHECK_NEAR(state.xnorm.data[2], 0.252732f, 1e-5f);
  CHECK(state.x.data[0] == t.token_embedding.data[288]); // x itself is unchanged

  ops_matvec(state.q, t.wq[0], state.xnorm);
  CHECK_NEAR(state.q.data[0], -0.432269f, 1e-5f);
  CHECK_NEAR(state.q.data[1], 0.146972f, 1e-5f);
  CHECK_NEAR(state.q.data[2], -0.345075f, 1e-5f);

  ops_matvec(state.k, t.wk[0], state.xnorm);
  CHECK_NEAR(state.k.data[0], 0.009520f, 1e-5f);
  CHECK_NEAR(state.k.data[1], -0.021181f, 1e-5f);
  CHECK_NEAR(state.k.data[2], 0.046356f, 1e-5f);

  ops_matvec(state.v, t.wv[0], state.xnorm);
  CHECK_NEAR(state.v.data[0], -0.013250f, 1e-5f);
  CHECK_NEAR(state.v.data[1], 0.038565f, 1e-5f);
  CHECK_NEAR(state.v.data[2], 0.030018f, 1e-5f);

  kv_store(&state, 0, 0); // layer 0, position 0 (BOS)
  CHECK(state.key_cache[0].data[0] == state.k.data[0]);
  CHECK(state.value_cache[0].data[287] == state.v.data[287]);
  CHECK(state.key_cache[0].data[288] == 0.0f); // position 1 still empty

  attention(&t, &state, 0, 0);
  CHECK_NEAR(state.att.data[0], 1.0f, 1e-6f);       // head 0: 100% on itself
  CHECK_NEAR(state.att.data[5 * 256], 1.0f, 1e-6f); // head 5
  for (int i = 0; i < 288; i++)
    CHECK(state.att_out.data[i] == state.v.data[i]); // one token: output = its own v

  ops_matvec(state.att_proj, t.wo[0], state.att_out);
  CHECK_NEAR(state.att_proj.data[0], 0.038377f, 1e-5f);
  CHECK_NEAR(state.att_proj.data[1], -0.006377f, 1e-5f);
  CHECK_NEAR(state.att_proj.data[2], 0.012379f, 1e-5f);

  run_state_free(&state);

  // Two tokens [BOS, " Once"], layer 0, with RoPE: matches the real model
  RunState s2 = run_state_create(c);
  embed(&t, &s2, 1);
  attention_block(&t, &s2, 0, 0); // BOS at position 0: leaves its k, v in the cache
  embed(&t, &s2, 9038);
  attention_block(&t, &s2, 0, 1);               // " Once" at position 1: looks at BOS and itself
  CHECK_NEAR(s2.att.data[0], 0.682737f, 1e-5f); // head 0: 68.3% on BOS
  CHECK_NEAR(s2.att.data[1 * 256], 0.998380f, 1e-5f); // head 1: 99.8% on BOS
  CHECK_NEAR(s2.att.data[3 * 256], 0.144991f, 1e-5f); // head 3: 14.5% on BOS
  CHECK_NEAR(s2.att.data[2 * 256], 0.621662f, 1e-5f); // head 2: 62.2%, differs most without RoPE
  CHECK_NEAR(s2.att.data[0] + s2.att.data[1], 1.0f, 1e-6f); // percentages sum to 1
  CHECK_NEAR(s2.att_out.data[0], 0.014552f, 1e-5f);         // mix of v
  run_state_free(&s2);

  // Full attention block for BOS: x = embedding + wo · attention
  RunState s3 = run_state_create(c);
  embed(&t, &s3, 1);
  attention_block(&t, &s3, 0, 0);
  CHECK_NEAR(s3.x.data[0], 0.021281f, 1e-5f);
  CHECK_NEAR(s3.x.data[1], -0.019908f, 1e-5f);
  CHECK_NEAR(s3.x.data[2], 0.019118f, 1e-5f);
  run_state_free(&s3);

  // Whole block 0 for BOS: attention + MLP
  RunState s4 = run_state_create(c);
  embed(&t, &s4, 1);
  attention_block(&t, &s4, 0, 0);
  mlp_block(&t, &s4, 0);
  CHECK_NEAR(s4.x.data[0], 0.038987f, 1e-5f);
  CHECK_NEAR(s4.x.data[1], 0.065597f, 1e-5f);
  CHECK_NEAR(s4.x.data[2], -0.080732f, 1e-5f);
  run_state_free(&s4);

  // Whole model for BOS: the most likely next token is " Once" (id 9038)
  RunState s5 = run_state_create(c);
  forward(&t, &s5, 1, 0);
  CHECK_NEAR(s5.logits.data[0], -6.7908f, 1e-3f);
  CHECK_NEAR(s5.logits.data[1], 0.8281f, 1e-3f);
  CHECK_NEAR(s5.logits.data[9038], 12.29f, 1e-2f);
  int best = 0;
  for (int i = 1; i < c.vocab_size; i++)
    if (s5.logits.data[i] > s5.logits.data[best])
      best = i;
  CHECK(best == 9038);
  run_state_free(&s5);
  free_transformer(&t);
  return report("test_transformer");
}
