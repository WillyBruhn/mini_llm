#include <torch/extension.h>

extern "C" {
#include "vector_addition.h"
}

torch::Tensor vector_addition(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have matching shapes");
    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");

    a = a.contiguous();
    b = b.contiguous();
    torch::Tensor out = torch::empty_like(a);
    size_t count = static_cast<size_t>(a.numel());

    if (a.is_cuda()) {
        vectoraddition_device(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            out.data_ptr<float>(),
            count
        );
    } else {
        vectoraddition(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            out.data_ptr<float>(),
            count
        );
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("vector_addition", &vector_addition, "Vector addition");
}
