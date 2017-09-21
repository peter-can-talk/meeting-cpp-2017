#include <cudnn.h>
#include <cstdlib>
#include <iostream>

int main(int argc, const char* argv[]) {
  int gpu_id = (argc > 1) ? std::atoi(argv[1]) : 0;
  std::cout << "GPU: " << gpu_id << std::endl;

  cudaSetDevice(gpu_id);

  cudnnHandle_t cudnn{nullptr};
  cudnnCreate(&cudnn);

  std::cout << "cudnn!" << std::endl;

  // float* d_data;
  // cudnnTensorDescriptor_t input_descriptor;
  // cudnnCreateTensorDescriptor(&input_descriptor);
  // cudnnSetTensor4dDescriptor(input_descriptor,
  //                            /*dataType=*/CUDNN_DATA_FLOAT,
  //                            /*format=*/CUDNN_TENSOR_NCHW,
  //                            /*batch_size=*/1,
  //                            /*channels=*/3,
  //                            /*image_height=*/5,
  //                            /*image_width=*/5);
  //
  // cudnnFilterDescriptor_t kernel_descriptor;
  // cudnnCreateFilterDescriptor(&kernel_descriptor);
  // cudnnSetTensor4dDescriptor(kernel_descriptor,
  //                            /*dataType=*/CUDNN_DATA_FLOAT,
  //                            /*format=*/CUDNN_TENSOR_NCHW,
  //                            /*out_channels=*/3,
  //                            /*in_channels=*/3,
  //                            /*kernel_height=*/5,
  //                            /*kernel_width=*/5);
  //
  // cudnnConvolutionDescriptor_t convolution_descriptor;
  // cudnnCreateConvolutionDescriptor(&convolution_descriptor);
  // cudnnSetConvolution2dDescriptor(convolution_descriptor,
  //                                 /*pad_height=*/0,
  //                                 /*pad_width=*/0,
  //                                 /*vertical_stride=*/1,
  //                                 /*horizontal_stride=*/1,
  //                                 /*dilation_height=*/1,
  //                                 /*dilation_width=*/1,
  //                                 /*mode=*/CUDNN_CROSS_CORRELATION,
  //                                 /*computeType=*/CUDNN_DATA_FLOAT);
  //
  // int n{0}, c{0}, h{0}, w{0};
  // cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
  //                                       input_descriptor,
  //                                       kernel_descriptor,
  //                                       &n,
  //                                       &c,
  //                                       &h,
  //                                       &w);
  //
  // std::cout << "n: " << n << "c: " << c << "h: " << h << "w: " << w
  //           << std::endl;
  //
  //
  // cudnnTensorDescriptor_t output_descriptor;
  // cudnnCreateTensorDescriptor(&input_descriptor);
  // cudnnSetTensor4dDescriptor(input_descriptor,
  //                            /*dataType=*/CUDNN_DATA_FLOAT,
  //                            /*format=*/CUDNN_TENSOR_NCHW,
  //                            /*batch_size=*/1,
  //                            /*channels=*/3,
  //                            /*image_height=*/5,
  //                            /*image_width=*/5);
  //
  // cudnnConvolutionFwdAlgo_t convolution_algorithm;
  // cudnnGetConvolutionForwardAlgorithm(cudnn,
  //                                     input_descriptor,
  //                                     kernel_descriptor,
  //                                     convolution_descriptor,
  //                                     output_descriptor,
  //                                     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
  //                                     0,
  //                                     &convolution_algorithm);
  // // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
  //
  // size_t workspace_bytes{0};
  // cudnnGetConvolutionForwardWorkspaceSize(cudnn,
  //                                         input_descriptor,
  //                                         kernel_descriptor,
  //                                         convolution_algorithm,
  //                                         output_descriptor,
  //                                         convolution_algorithm,
  //                                         &workspace_bytes);
  // std::cout << "Workspace size: " << workspace_bytes << "B" << std::endl;
  //
  // cudnnConvolutionForward(cudnn,
  //                         &alpha,
  //                         &beta,
  //                         input_descriptor,
  //                         ___image___,
  //                         kernel_descriptor,
  //                         ___kernel___,
  //                         convolution_descriptor,
  //                         convolution_algorithm,
  //                         d_workspace,
  //                         workspace_bytes,
  //                         output_descriptor,
  //                         ___output___);
  //
  // cudnnActivationDescriptor_t activation_descriptor;
  // cudnnCreateActivationDescriptor(&activation_descriptor,
  //                                 CUDNN_ACTIVATION_SIGMOID,
  //                                 CUDNN_PROPAGATE_NAN,
  //                                 /*relu_coef=*/0);
  // cudnnActivationForward(cudnn,
  //                        activation_descriptor,
  //                        &alpha,
  //                        output_descriptor,
  //                        ___output___,
  //                        &beta,
  //                        output_descriptor,
  //                        ___output___);


  cudnnDestroy(cudnn);
}
