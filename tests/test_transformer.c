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

// Two identical rows rotated from pos0 = 3: row 0 = rope alone at 3, row 1 = rope alone at 4
static void test_rope_rows(void) {
  float v[] = {1, 2, 3, 4};
  Matrix q2 = mat_create(2, 4), k2 = mat_create(2, 4), q1 = mat_create(1, 4), k1 = mat_create(1, 4);
  for (int i = 0; i < 8; i++)
    q2.data[i] = k2.data[i] = v[i % 4];
  rope(q2, k2, 3, 4);
  for (int pos = 3; pos <= 4; pos++) {
    for (int i = 0; i < 4; i++)
      q1.data[i] = k1.data[i] = v[i];
    rope(q1, k1, pos, 4);
    for (int i = 0; i < 4; i++) {
      CHECK(q2.data[(pos - 3) * 4 + i] == q1.data[i]);
      CHECK(k2.data[(pos - 3) * 4 + i] == k1.data[i]);
    }
  }
  CHECK(q2.data[0] != q2.data[4]); // different positions really got different angles
  mat_free(q2);
  mat_free(k2);
  mat_free(q1);
  mat_free(k1);
}

// Expected values were read straight from the file with `od` (see notes).
int main(void) {
  test_rmsnorm_formula();
  test_silu();
  test_rope_rows();

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

  int bos = 1, once = 9038;
  RunState state = run_state_create(c);
  embed(&t, &state, &bos, 1);

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

  ops_linear(state.q, state.xnorm, t.wq[0]);
  CHECK_NEAR(state.q.data[0], -0.432269f, 1e-5f);
  CHECK_NEAR(state.q.data[1], 0.146972f, 1e-5f);
  CHECK_NEAR(state.q.data[2], -0.345075f, 1e-5f);

  ops_linear(state.k, state.xnorm, t.wk[0]);
  CHECK_NEAR(state.k.data[0], 0.009520f, 1e-5f);
  CHECK_NEAR(state.k.data[1], -0.021181f, 1e-5f);
  CHECK_NEAR(state.k.data[2], 0.046356f, 1e-5f);

  ops_linear(state.v, state.xnorm, t.wv[0]);
  CHECK_NEAR(state.v.data[0], -0.013250f, 1e-5f);
  CHECK_NEAR(state.v.data[1], 0.038565f, 1e-5f);
  CHECK_NEAR(state.v.data[2], 0.030018f, 1e-5f);

  kv_store(&state, 0, 0); // layer 0, position 0 (BOS)
  CHECK(state.key_cache[0].data[0] == state.k.data[0]);
  CHECK(state.value_cache[0].data[287] == state.v.data[287]);
  CHECK(state.key_cache[0].data[288] == 0.0f); // position 1 still empty

  // kv_store of 2 rows from pos0 = 3: cache rows 3, 4 = k, v rows 0, 1; rows 2 and 5 untouched
  RunState skv = run_state_create(c);
  skv.k.rows = skv.v.rows = 2;
  for (int i = 0; i < 2 * 288; i++) {
    skv.k.data[i] = i + 1;
    skv.v.data[i] = -(i + 1);
  }
  kv_store(&skv, 0, 3);
  for (int i = 0; i < 2 * 288; i++) {
    CHECK(skv.key_cache[0].data[3 * 288 + i] == skv.k.data[i]);
    CHECK(skv.value_cache[0].data[3 * 288 + i] == skv.v.data[i]);
  }
  CHECK(skv.key_cache[0].data[3 * 288 - 1] == 0.0f);
  CHECK(skv.key_cache[0].data[5 * 288] == 0.0f);
  run_state_free(&skv);

  attention(&t, &state, 0, 0);
  CHECK_NEAR(state.att.data[0], 1.0f, 1e-6f);       // head 0: 100% on itself
  CHECK_NEAR(state.att.data[5 * 256], 1.0f, 1e-6f); // head 5
  for (int i = 0; i < 288; i++)
    CHECK(state.att_out.data[i] == state.v.data[i]); // one token: output = its own v

  ops_linear(state.att_proj, state.att_out, t.wo[0]);
  CHECK_NEAR(state.att_proj.data[0], 0.038377f, 1e-5f);
  CHECK_NEAR(state.att_proj.data[1], -0.006377f, 1e-5f);
  CHECK_NEAR(state.att_proj.data[2], 0.012379f, 1e-5f);

  run_state_free(&state);

  // Two tokens [BOS, " Once"], layer 0, with RoPE: matches the real model
  RunState s2 = run_state_create(c);
  embed(&t, &s2, &bos, 1);
  attention_block(&t, &s2, 0, 0); // BOS at position 0: leaves its k, v in the cache
  embed(&t, &s2, &once, 1);
  attention_block(&t, &s2, 0, 1);               // " Once" at position 1: looks at BOS and itself
  CHECK_NEAR(s2.att.data[0], 0.682737f, 1e-5f); // head 0: 68.3% on BOS
  CHECK_NEAR(s2.att.data[1 * 256], 0.998380f, 1e-5f); // head 1: 99.8% on BOS
  CHECK_NEAR(s2.att.data[3 * 256], 0.144991f, 1e-5f); // head 3: 14.5% on BOS
  CHECK_NEAR(s2.att.data[2 * 256], 0.621662f, 1e-5f); // head 2: 62.2%, differs most without RoPE
  CHECK_NEAR(s2.att.data[0] + s2.att.data[1], 1.0f, 1e-6f); // percentages sum to 1
  CHECK_NEAR(s2.att_out.data[0], 0.014552f, 1e-5f);         // mix of v
  run_state_free(&s2);

  // The same two tokens at once (n = 2): row 1 = " Once" above, row 0 = BOS alone (s3 below)
  RunState sb = run_state_create(c);
  int two[] = {1, 9038};
  set_rows(&sb, 2);
  embed(&t, &sb, two, 2);
  attention_block(&t, &sb, 0, 0);
  CHECK_NEAR(sb.att.data[0], 0.682737f, 1e-5f); // att holds the last row: " Once"
  CHECK_NEAR(sb.att.data[1 * 256], 0.998380f, 1e-5f);
  CHECK_NEAR(sb.att_out.data[288], 0.014552f, 1e-5f); // row 1 of att_out
  CHECK_NEAR(sb.x.data[0], 0.021281f, 1e-5f);         // row 0: BOS must not see " Once"
  CHECK_NEAR(sb.x.data[1], -0.019908f, 1e-5f);
  run_state_free(&sb);

  // Full attention block for BOS: x = embedding + wo · attention
  RunState s3 = run_state_create(c);
  embed(&t, &s3, &bos, 1);
  attention_block(&t, &s3, 0, 0);
  CHECK_NEAR(s3.x.data[0], 0.021281f, 1e-5f);
  CHECK_NEAR(s3.x.data[1], -0.019908f, 1e-5f);
  CHECK_NEAR(s3.x.data[2], 0.019118f, 1e-5f);
  run_state_free(&s3);

  // Whole block 0 for BOS: attention + MLP
  RunState s4 = run_state_create(c);
  embed(&t, &s4, &bos, 1);
  attention_block(&t, &s4, 0, 0);
  mlp_block(&t, &s4, 0);
  CHECK_NEAR(s4.x.data[0], 0.038987f, 1e-5f);
  CHECK_NEAR(s4.x.data[1], 0.065597f, 1e-5f);
  CHECK_NEAR(s4.x.data[2], -0.080732f, 1e-5f);
  run_state_free(&s4);

  // Whole model for BOS: the most likely next token is " Once" (id 9038)
  RunState s5 = run_state_create(c);
  forward(&t, &s5, &bos, 1, 0);
  CHECK_NEAR(s5.logits.data[0], -6.7908f, 1e-3f);
  CHECK_NEAR(s5.logits.data[1], 0.8281f, 1e-3f);
  CHECK_NEAR(s5.logits.data[9038], 12.29f, 1e-2f);
  int best = 0;
  for (int i = 1; i < c.vocab_size; i++)
    if (s5.logits.data[i] > s5.logits.data[best])
      best = i;
  CHECK(best == 9038);
  run_state_free(&s5);

  // Greedy generation from BOS matches an independent Python implementation:
  // " Once upon a time, there was a little girl named L"
  int expected[] = {9038, 2501, 263, 931, 29892, 727, 471, 263, 2217, 7826, 4257, 365};
  RunState s6 = run_state_create(c);
  int token = 1;
  for (int pos = 0; pos < 12; pos++) {
    forward(&t, &s6, &token, 1, pos);
    token = argmax(s6.logits);
    CHECK(token == expected[pos]);
  }
  run_state_free(&s6);

  // Prefill at once == prefill one token at a time: same logits, same KV cache
  int prompt[] = {1, 450, 8338, 265, 471}; // BOS " The" " drag" "on" " was"
  RunState one = run_state_create(c), all = run_state_create(c);
  for (int pos = 0; pos < 5; pos++)
    forward(&t, &one, &prompt[pos], 1, pos);
  prefill(&t, &all, prompt, 5);
  for (int i = 0; i < c.vocab_size; i++)
    CHECK(all.logits.data[i] == one.logits.data[i]);
  for (int l = 0; l < c.n_layers; l++)
    for (int i = 0; i < 5 * c.dim; i++) {
      CHECK(all.key_cache[l].data[i] == one.key_cache[l].data[i]);
      CHECK(all.value_cache[l].data[i] == one.value_cache[l].data[i]);
    }
  run_state_free(&one);
  run_state_free(&all);
  free_transformer(&t);
  return report("test_transformer");
}
