#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

namespace tensorflow {

REGISTER_OP("CppConSigmoid")
    .Attr("T: {float, double, int32}")
    .Input("tensor: T")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      context->set_output(0, context->input(0));
      return Status::OK();
    });

template <typename T>
class CppConSigmoid : public OpKernel {
 public:
  explicit CppConSigmoid(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const auto input = input_tensor.flat<T>();

    Tensor* output_tensor{nullptr};
    OP_REQUIRES_OK(context,
                   context->allocate_output(0,
                                            input_tensor.shape(),
                                            &output_tensor));
    auto output = output_tensor->flat<T>();

    // sigmoid(z) = 1 / (1 + exp(-z))
    output = (1 + (-input).exp()).inverse();
  }
};

#define CPU_KERNEL(T)                                  \
  REGISTER_KERNEL_BUILDER(Name("CppConSigmoid")        \
                              .Device(DEVICE_CPU)      \
                              .TypeConstraint<T>("T"), \
                          CppConSigmoid<T>);

CPU_KERNEL(int32)
CPU_KERNEL(float)
CPU_KERNEL(double)
}  // namespace tensorflow
