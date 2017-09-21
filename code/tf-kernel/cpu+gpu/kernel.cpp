#define EIGEN_USE_THREADS

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "kernel.cuh"
#endif  // GOOGLE_CUDA

template <typename T>
struct CPUKernel {
  void operator()(tensorflow::OpKernelContext*,
                  const tensorflow::Tensor& input_tensor,
                  tensorflow::Tensor& output_tensor) {
    auto input = input_tensor.flat<T>();
    auto output = output_tensor.flat<T>();
    output = (1 + (-input).exp()).inverse();
  }
};

namespace tensorflow {

REGISTER_OP("CppConSigmoid")
    .Attr("T: {float, double, int32}")
    .Input("tensor: T")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      context->set_output(0, context->input(0));
      return Status::OK();
    });

template <template <typename T> class Kernel, typename T>
class CppConSigmoid : public OpKernel {
 public:
  explicit CppConSigmoid(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);

    Tensor* output_tensor{nullptr};
    OP_REQUIRES_OK(context,
                   context->allocate_output(0,
                                            input_tensor.shape(),
                                            &output_tensor));

    Kernel<T>()(context, input_tensor, *output_tensor);
  }
};

#define CPU_KERNEL(T)                                  \
  REGISTER_KERNEL_BUILDER(Name("CppConSigmoid")        \
                              .Device(DEVICE_CPU)      \
                              .TypeConstraint<T>("T"), \
                          CppConSigmoid<CPUKernel, T>);

#define GPU_KERNEL(T)                                  \
  REGISTER_KERNEL_BUILDER(Name("CppConSigmoid")        \
                              .Device(DEVICE_GPU)      \
                              .TypeConstraint<T>("T"), \
                          CppConSigmoid<GPUKernel, T>);

CPU_KERNEL(int32)
CPU_KERNEL(float)
CPU_KERNEL(double)

#if GOOGLE_CUDA
GPU_KERNEL(int32)
GPU_KERNEL(float)
GPU_KERNEL(double)
#endif  // GOOGLE_CUDA
}  // namespace tensorflow
