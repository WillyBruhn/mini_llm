#include "../include/attention.h"
#include "../include/layernorm.h"
#include "../include/matrix_mult.h"
#include "../include/transformer_block.h"

// The decode block of transformer_block.cu for several sequences. The launches are the same
// (7 per layer); only the kernels know about the rows.
void transformer_block_decode_batch_device(const float *d_input,
                                           float *d_out,
                                           const TransformerBlockWeights *w,
                                           const TransformerBlockHalfWeights *half,
                                           const TransformerBlockScratch *s,
                                           float *const *d_cache_k,
                                           float *const *d_cache_v,
                                           const int *d_lengths,
                                           size_t rows,
                                           size_t max_length,
                                           size_t hidden_size,
                                           size_t n_heads,
                                           size_t mlp_size,
                                           float epsilon) {
    int use_half = half != NULL;

    layernorm_device(s->ln1, d_input, w->ln1_gamma, w->ln1_beta, rows, hidden_size, epsilon);

    // q, k and v in one launch. K and V go to the scratch rows, and the attention kernel stores
    // them in the caches (every sequence has its own cache row).
    MatvecProduct qkv[3] = {
        {use_half ? NULL : w->query_weights, use_half ? half->query_weights : NULL, w->query_bias,
         NULL, s->q},
        {use_half ? NULL : w->key_weights, use_half ? half->key_weights : NULL, w->key_bias, NULL,
         s->k},
        {use_half ? NULL : w->value_weights, use_half ? half->value_weights : NULL, w->value_bias,
         NULL, s->v},
    };
    matvec_batch_device(s->ln1, qkv, 3, MATVEC_EPILOGUE_BIAS, rows, hidden_size, hidden_size);

    decode_attention_batch_device(s->q, s->k, s->v, d_cache_k, d_cache_v, d_lengths, s->attention,
                                  rows, max_length, n_heads, hidden_size / n_heads);

    MatvecProduct projection = {use_half ? NULL : w->attention_weights,
                                use_half ? half->attention_weights : NULL, w->attention_bias,
                                d_input, s->residual};
    matvec_batch_device(s->attention, &projection, 1, MATVEC_EPILOGUE_BIAS_RESIDUAL, rows,
                        hidden_size, hidden_size);

    layernorm_device(s->ln2, s->residual, w->ln2_gamma, w->ln2_beta, rows, hidden_size, epsilon);

    MatvecProduct up = {use_half ? NULL : w->mlp_up_weights, use_half ? half->mlp_up_weights : NULL,
                        w->mlp_up_bias, NULL, s->mlp_activated};
    matvec_batch_device(s->ln2, &up, 1, MATVEC_EPILOGUE_BIAS_GELU, rows, hidden_size, mlp_size);

    MatvecProduct down = {use_half ? NULL : w->mlp_down_weights,
                          use_half ? half->mlp_down_weights : NULL, w->mlp_down_bias, s->residual,
                          d_out};
    matvec_batch_device(s->mlp_activated, &down, 1, MATVEC_EPILOGUE_BIAS_RESIDUAL, rows, mlp_size,
                        hidden_size);
}
