#ifndef KERNEL_CUH
#define KERNEL_CUH

namespace tensorflow {
class OpKernelContext;
class Tensor;
}  // namespace tensorflow

template <typename T>
struct GPUKernel {
  void operator()(tensorflow::OpKernelContext* context,
                  const tensorflow::Tensor& input_tensor,
                  tensorflow::Tensor& output_tensor);
};

#endif  // KERNEL_CUH
