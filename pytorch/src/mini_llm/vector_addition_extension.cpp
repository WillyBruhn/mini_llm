#include <torch/extension.h>

extern "C" {
#include "attention.h"
#include "embedding.h"
#include "gelu.h"
#include "layernorm.h"
#include "linear.h"
#include "softmax.h"
#include "transformer_block.h"
#include "vector_addition.h"
}

void check_cpu_float(const torch::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
}

torch::Tensor vector_addition(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have matching shapes");
    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");
    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");

    a = a.contiguous();
    b = b.contiguous();
    torch::Tensor out = torch::empty_like(a);
    size_t count = static_cast<size_t>(a.numel());

    if (a.is_cuda()) {
        vectoraddition_device(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
                              count);
    } else {
        vectoraddition(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), count);
    }

    return out;
}

torch::Tensor
embedding(torch::Tensor tokens, torch::Tensor token_embedding, torch::Tensor position_embedding) {
    check_cpu_float(token_embedding, "token_embedding");
    check_cpu_float(position_embedding, "position_embedding");
    TORCH_CHECK(tokens.device().is_cpu() && tokens.scalar_type() == torch::kInt32,
                "tokens must be a CPU int32 tensor");
    tokens = tokens.contiguous();
    token_embedding = token_embedding.contiguous();
    position_embedding = position_embedding.contiguous();
    TORCH_CHECK(tokens.dim() == 1, "tokens must have shape [sequence]");
    int64_t sequence_length = tokens.numel();
    int64_t hidden_size = token_embedding.size(1);
    auto out = torch::empty({sequence_length, hidden_size}, token_embedding.options());
    embed_tokens(out.data_ptr<float>(), tokens.data_ptr<int>(), token_embedding.data_ptr<float>(),
                 position_embedding.data_ptr<float>(), sequence_length, hidden_size);
    return out;
}

torch::Tensor
layer_norm(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, double epsilon) {
    check_cpu_float(input, "input");
    check_cpu_float(gamma, "gamma");
    check_cpu_float(beta, "beta");
    input = input.contiguous();
    gamma = gamma.contiguous();
    beta = beta.contiguous();
    TORCH_CHECK(input.dim() == 2, "input must have shape [rows, columns]");
    auto out = torch::empty_like(input);
    layernorm(out.data_ptr<float>(), input.data_ptr<float>(), gamma.data_ptr<float>(),
              beta.data_ptr<float>(), input.size(0), input.size(1), (float)epsilon);
    return out;
}

torch::Tensor gelu_new_op(torch::Tensor input) {
    check_cpu_float(input, "input");
    input = input.contiguous();
    auto out = torch::empty_like(input);
    ::gelu_new(out.data_ptr<float>(), input.data_ptr<float>(), input.numel());
    return out;
}

torch::Tensor linear_layer(torch::Tensor input, torch::Tensor weights, torch::Tensor bias) {
    check_cpu_float(input, "input");
    check_cpu_float(weights, "weights");
    check_cpu_float(bias, "bias");
    input = input.contiguous();
    weights = weights.contiguous();
    bias = bias.contiguous();
    TORCH_CHECK(input.dim() == 2, "input must have shape [rows, input_columns]");
    auto out = torch::empty({input.size(0), weights.size(1)}, input.options());
    linear(input.data_ptr<float>(), weights.data_ptr<float>(), bias.data_ptr<float>(),
           out.data_ptr<float>(), input.size(0), input.size(1), weights.size(1));
    return out;
}

torch::Tensor softmax_rows(torch::Tensor input, bool causal) {
    check_cpu_float(input, "input");
    input = input.contiguous();
    TORCH_CHECK(input.dim() == 2, "input must have shape [rows, columns]");
    auto out = torch::empty_like(input);
    if (causal) {
        causal_softmax(input.data_ptr<float>(), out.data_ptr<float>(), input.size(0),
                       input.size(1));
    } else {
        softmax(input.data_ptr<float>(), out.data_ptr<float>(), input.size(0), input.size(1));
    }
    return out;
}

torch::Tensor
attention_head_cpu(torch::Tensor query, torch::Tensor key, torch::Tensor value, int64_t n_heads) {
    check_cpu_float(query, "query");
    check_cpu_float(key, "key");
    check_cpu_float(value, "value");
    query = query.contiguous();
    key = key.contiguous();
    value = value.contiguous();
    TORCH_CHECK(query.sizes() == key.sizes() && query.sizes() == value.sizes(),
                "query, key, and value must have matching shapes");
    TORCH_CHECK(query.dim() == 2 && query.size(1) % n_heads == 0,
                "hidden size must divide evenly by n_heads");
    auto out = torch::empty_like(query);
    multi_head_attention(query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
                         out.data_ptr<float>(), query.size(0), n_heads, query.size(1) / n_heads);
    return out;
}

torch::Tensor transformer_block(torch::Tensor input,
                                torch::Tensor ln1_gamma,
                                torch::Tensor ln1_beta,
                                torch::Tensor query_weights,
                                torch::Tensor query_bias,
                                torch::Tensor key_weights,
                                torch::Tensor key_bias,
                                torch::Tensor value_weights,
                                torch::Tensor value_bias,
                                torch::Tensor attention_weights,
                                torch::Tensor attention_bias,
                                torch::Tensor ln2_gamma,
                                torch::Tensor ln2_beta,
                                torch::Tensor mlp_up_weights,
                                torch::Tensor mlp_up_bias,
                                torch::Tensor mlp_down_weights,
                                torch::Tensor mlp_down_bias,
                                int64_t n_heads,
                                int64_t mlp_size,
                                double epsilon) {
    torch::Tensor tensors[] = {
        input,    ln1_gamma,      ln1_beta,    query_weights,     query_bias,     key_weights,
        key_bias, value_weights,  value_bias,  attention_weights, attention_bias, ln2_gamma,
        ln2_beta, mlp_up_weights, mlp_up_bias, mlp_down_weights,  mlp_down_bias,
    };
    for (const torch::Tensor &tensor : tensors) {
        TORCH_CHECK(tensor.device().is_cpu(), "transformer_block is CPU-only");
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
                    "transformer_block requires float32 tensors");
    }

    TORCH_CHECK(input.dim() == 2, "input must have shape [sequence, hidden]");
    input = input.contiguous();
    ln1_gamma = ln1_gamma.contiguous();
    ln1_beta = ln1_beta.contiguous();
    query_weights = query_weights.contiguous();
    query_bias = query_bias.contiguous();
    key_weights = key_weights.contiguous();
    key_bias = key_bias.contiguous();
    value_weights = value_weights.contiguous();
    value_bias = value_bias.contiguous();
    attention_weights = attention_weights.contiguous();
    attention_bias = attention_bias.contiguous();
    ln2_gamma = ln2_gamma.contiguous();
    ln2_beta = ln2_beta.contiguous();
    mlp_up_weights = mlp_up_weights.contiguous();
    mlp_up_bias = mlp_up_bias.contiguous();
    mlp_down_weights = mlp_down_weights.contiguous();
    mlp_down_bias = mlp_down_bias.contiguous();

    int64_t sequence_length = input.size(0);
    int64_t hidden_size = input.size(1);
    TORCH_CHECK(hidden_size % n_heads == 0, "hidden size must divide evenly by heads");
    TORCH_CHECK(mlp_up_weights.size(1) == mlp_size, "mlp_size does not match mlp_up_weights");

    torch::Tensor out = torch::empty_like(input);
    TransformerBlockWeights weights = {
        ln1_gamma.data_ptr<float>(),         ln1_beta.data_ptr<float>(),
        query_weights.data_ptr<float>(),     query_bias.data_ptr<float>(),
        key_weights.data_ptr<float>(),       key_bias.data_ptr<float>(),
        value_weights.data_ptr<float>(),     value_bias.data_ptr<float>(),
        attention_weights.data_ptr<float>(), attention_bias.data_ptr<float>(),
        ln2_gamma.data_ptr<float>(),         ln2_beta.data_ptr<float>(),
        mlp_up_weights.data_ptr<float>(),    mlp_up_bias.data_ptr<float>(),
        mlp_down_weights.data_ptr<float>(),  mlp_down_bias.data_ptr<float>(),
    };

    transformer_block_forward(input.data_ptr<float>(), out.data_ptr<float>(), &weights,
                              static_cast<size_t>(sequence_length),
                              static_cast<size_t>(hidden_size), static_cast<size_t>(n_heads),
                              static_cast<size_t>(mlp_size), static_cast<float>(epsilon));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("vector_addition", &vector_addition, "Vector addition");
    module.def("embedding", &embedding, "Token and position embedding");
    module.def("layer_norm", &layer_norm, "Layer normalization");
    module.def("gelu_new", &gelu_new_op, "GPT-2 GELU approximation");
    module.def("linear", &linear_layer, "Linear projection");
    module.def(
        "softmax", [](torch::Tensor input) { return softmax_rows(input, false); },
        "Row-wise softmax");
    module.def(
        "causal_softmax", [](torch::Tensor input) { return softmax_rows(input, true); },
        "Causal row-wise softmax");
    module.def("attention", &attention_head_cpu, "Multi-head causal attention");
    module.def("transformer_block", &transformer_block, "CPU transformer block");
}
