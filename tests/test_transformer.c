#include "check.h"
#include "transformer.h"

// Expected values were read straight from the file with `od` (see notes).
int main(void) {
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

  run_state_free(&state);
  free_transformer(&t);
  return report("test_transformer");
}
