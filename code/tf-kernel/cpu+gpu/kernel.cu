#define EIGEN_USE_GPU

#include "kernel.cuh"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/tensor.h>

template <typename T>
__global__ void sigmoid_kernel(const T* __restrict__ input,
                               T* __restrict__ output,
                               const long size) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;
  output[index] = 1 / (1 + __expf(-__ldg(&input[index])));
}

template <typename T>
void GPUKernel<T>::operator()(tensorflow::OpKernelContext* context,
                              const tensorflow::Tensor& input_tensor,
                              tensorflow::Tensor& output_tensor) {
  const auto& device = context->eigen_device<Eigen::GpuDevice>();

  auto input = input_tensor.flat<T>();
  auto output = output_tensor.flat<T>();

  const int blocks = 1024;
  const int threads = (input.size() + blocks - 1) / blocks;
  sigmoid_kernel<T><<<blocks, threads, 0, device.stream()>>>(input.data(),
                                                             output.data(),
                                                             input.size());
}


template struct GPUKernel<float>;
template struct GPUKernel<double>;
