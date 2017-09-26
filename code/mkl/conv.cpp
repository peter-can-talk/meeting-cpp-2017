#include <mkl_dnn.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#define checkMKL(expression)                                      \
  {                                                               \
    dnnError_t status = expression;                               \
    if (status != E_SUCCESS) {                                    \
      std::cerr << "Error at line " << __LINE__ << ": " << status \
                << std::endl;                                     \
      std::exit(EXIT_FAILURE);                                    \
    }                                                             \
  }

cv::Mat load_image(const char* image_path, bool is_gray) {
  cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
  image.convertTo(image, CV_32FC3);
  if (is_gray) {
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
  }
  cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
  std::cout << "Input Image: " << image.rows << " x " << image.cols << " x "
            << image.channels() << std::endl;
  return image;
}

std::vector<float>
flip_channels(float* input_buffer, size_t X, size_t Y, size_t Z) {
  assert(input_buffer != nullptr);
  std::vector<float> flipped(X * Y * Z);

  for (size_t x = 0; x < X; ++x) {
    for (size_t y = 0; y < Y; ++y) {
      for (size_t z = 0; z < Z; ++z) {
        const size_t index = x * (Y * Z) + y * Z + z;
        const size_t flipped_index = z * (X * Y) + y * X + x;
        assert(flipped_index < X * Y * Z);
        flipped[flipped_index] = input_buffer[index];
      }
    }
  }

  return flipped;
}


void save_image(const char* output_filename,
                float* buffer,
                int height,
                int width,
                bool is_gray) {
  const auto format = is_gray ? CV_32F : CV_32FC3;
  cv::Mat output_image(height, width, format, buffer);
  // Make negative values zero.
  cv::threshold(output_image,
                output_image,
                /*threshold=*/0,
                /*maxval=*/0,
                cv::THRESH_TOZERO);
  cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
  output_image.convertTo(output_image, CV_8UC3);
  cv::imwrite(output_filename, output_image);
  std::cerr << "Wrote output to " << output_filename << std::endl;
}

void setup_conversion(dnnPrimitive_t* conversion_primitive,
                      dnnLayout_t source_layout,
                      dnnLayout_t target_layout,
                      float* source_buffer,
                      float** target_buffer) {
  if (!dnnLayoutCompare_F32(target_layout, source_layout)) {
    checkMKL(dnnConversionCreate_F32(conversion_primitive,
                                     source_layout,
                                     target_layout));
    checkMKL(dnnAllocateBuffer_F32(reinterpret_cast<void**>(target_buffer),
                                   target_layout));
  } else {
    assert(source_buffer != nullptr);
    *target_buffer = source_buffer;
  }

  assert(target_buffer != nullptr);
}

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    std::cerr << "usage: conv <image> [is_gray]" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  bool is_gray = false;
  if (argc == 3) {
    is_gray = std::atoi(argv[2]);
  }

  cv::Mat image = load_image(argv[1], is_gray);

  const size_t height = image.rows;
  const size_t width = image.cols;
  const size_t input_channels = is_gray ? 1 : 3;
  const size_t output_channels = is_gray ? 1 : 3;
  const size_t batch_size = 1;
  const int kernel_size = 5;
  const size_t dimension = 4;

  // Format is: WHCN
  size_t input_shape[] = {width, height, input_channels, batch_size};
  size_t input_strides[] = {1,
                            width,
                            width * height,
                            width * height * input_channels};

  size_t output_shape[] = {width, height, output_channels, batch_size};
  size_t output_strides[] = {1,
                             width,
                             width * height,
                             width * height * output_channels};

  // HWIO
  size_t kernel_shape[] = {kernel_size,
                           kernel_size,
                           input_channels,
                           output_channels};
  size_t kernel_strides[] = {1,
                             kernel_size,
                             kernel_size * kernel_size,
                             kernel_size * kernel_size * input_channels};

  dnnLayout_t input_layout{nullptr};
  checkMKL(dnnLayoutCreate_F32(&input_layout,
                               dimension,
                               input_shape,
                               input_strides));

  dnnLayout_t output_layout{nullptr};
  checkMKL(dnnLayoutCreate_F32(&output_layout,
                               dimension,
                               output_shape,
                               output_strides));

  dnnLayout_t kernel_layout{nullptr};
  checkMKL(dnnLayoutCreate_F32(&kernel_layout,
                               dimension,
                               kernel_shape,
                               kernel_strides));

  // assert(dnnLayoutCompare_F32(input_layout, output_layout));

  dnnPrimitiveAttributes_t attributes{nullptr};
  checkMKL(dnnPrimitiveAttributesCreate_F32(&attributes));

  size_t convolution_strides[] = {/*width=*/1, /*height=*/1};
  int convolution_offsets[] = {/*horizontal=*/(1 - kernel_size) / 2,
                               /*vertical=*/(1 - kernel_size) / 2};

  dnnPrimitive_t convolution_primitive{nullptr};
  checkMKL(dnnConvolutionCreateForward_F32(&convolution_primitive,
                                           attributes,
                                           dnnAlgorithmConvolutionDirect,
                                           dimension,
                                           input_shape,
                                           output_shape,
                                           kernel_shape,
                                           convolution_strides,
                                           convolution_offsets,
                                           dnnBorderZeros));

  dnnLayout_t conv_input_layout{nullptr};
  checkMKL(dnnLayoutCreateFromPrimitive_F32(&conv_input_layout,
                                            convolution_primitive,
                                            dnnResourceSrc));
  std::cerr << "Input size: " << dnnLayoutGetMemorySize_F32(conv_input_layout)
            << "B" << std::endl;


  dnnLayout_t conv_output_layout{nullptr};
  checkMKL(dnnLayoutCreateFromPrimitive_F32(&conv_output_layout,
                                            convolution_primitive,
                                            dnnResourceDst));
  std::cerr << "Output size: " << dnnLayoutGetMemorySize_F32(conv_output_layout)
            << "B" << std::endl;

  dnnLayout_t conv_kernel_layout{nullptr};
  checkMKL(dnnLayoutCreateFromPrimitive_F32(&conv_kernel_layout,
                                            convolution_primitive,
                                            dnnResourceFilter));
  std::cerr << "Kernel size: " << dnnLayoutGetMemorySize_F32(conv_kernel_layout)
            << "B" << std::endl;

  auto input_buffer =
      flip_channels(image.ptr<float>(0), height, width, input_channels);
  float* output_buffer{nullptr};
  float* conversion_buffer[dnnResourceNumber] = {nullptr};

  // clang-format off
  float kernel_template[kernel_size][kernel_size] = {
    {-1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1},
    {-1, -1, 24, -1, -1},
    {-1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1},
  };
  // clang-format on

  float kernel_buffer[output_channels][input_channels][kernel_size]
                     [kernel_size];
  for (size_t output_channel = 0; output_channel < output_channels;
       ++output_channel) {
    for (size_t input_channel = 0; input_channel < input_channels;
         ++input_channel) {
      for (size_t column = 0; column < kernel_size; ++column) {
        for (size_t row = 0; row < kernel_size; ++row) {
          kernel_buffer[output_channel][input_channel][column][row] =
              kernel_template[column][row];
        }
      }
    }
  }
  float* kernel_buffer_flat = &kernel_buffer[0][0][0][0];

  dnnPrimitive_t input_conversion{nullptr};
  setup_conversion(&input_conversion,
                   input_layout,
                   conv_input_layout,
                   input_buffer.data(),
                   &conversion_buffer[dnnResourceSrc]);

  dnnPrimitive_t kernel_conversion{nullptr};
  setup_conversion(&kernel_conversion,
                   kernel_layout,
                   conv_kernel_layout,
                   kernel_buffer_flat,
                   &conversion_buffer[dnnResourceFilter]);

  checkMKL(dnnAllocateBuffer_F32(reinterpret_cast<void**>(
                                     &conversion_buffer[dnnResourceDst]),
                                 conv_output_layout));

  dnnPrimitive_t output_conversion{nullptr};
  setup_conversion(&output_conversion,
                   conv_output_layout,
                   output_layout,
                   conversion_buffer[dnnResourceDst],
                   &output_buffer);

  if (kernel_conversion) {
    std::cerr << "Performing kernel conversion" << std::endl;
    checkMKL(dnnConversionExecute_F32(kernel_conversion,
                                      kernel_buffer,
                                      conversion_buffer[dnnResourceFilter]));
  } else {
    std::cerr << "Skipping kernel conversion" << std::endl;
  }

  if (input_conversion) {
    std::cerr << "Performing input conversion" << std::endl;
    checkMKL(dnnConversionExecute_F32(input_conversion,
                                      input_buffer.data(),
                                      conversion_buffer[dnnResourceSrc]));
  } else {
    std::cerr << "Skipping input conversion" << std::endl;
  }

  std::cerr << "Executing convolution" << std::endl;
  checkMKL(dnnExecute_F32(convolution_primitive,
                          reinterpret_cast<void**>(conversion_buffer)));


  if (output_conversion) {
    std::cerr << "Performing output conversion" << std::endl;
    checkMKL(dnnConversionExecute_F32(output_conversion,
                                      conversion_buffer[dnnResourceDst],
                                      output_buffer));
  } else {
    std::cerr << "Skipping output conversion" << std::endl;
  }

  auto flipped_output =
      flip_channels(output_buffer, output_channels, height, width);
  save_image("mkl-out.png", flipped_output.data(), height, width, is_gray);

  // ---------------------------------------------------------------------------

  checkMKL(dnnPrimitiveAttributesDestroy_F32(attributes));

  checkMKL(dnnLayoutDelete_F32(kernel_layout));
  checkMKL(dnnLayoutDelete_F32(output_layout));
  checkMKL(dnnLayoutDelete_F32(input_layout));

  checkMKL(dnnLayoutDelete_F32(conv_kernel_layout));
  checkMKL(dnnLayoutDelete_F32(conv_output_layout));
  checkMKL(dnnLayoutDelete_F32(conv_input_layout));

  if (conversion_buffer[dnnResourceSrc] != input_buffer.data()) {
    checkMKL(dnnReleaseBuffer_F32(conversion_buffer[dnnResourceSrc]));
  }
  if (conversion_buffer[dnnResourceFilter] != kernel_buffer_flat) {
    checkMKL(dnnReleaseBuffer_F32(conversion_buffer[dnnResourceFilter]));
  }
  checkMKL(dnnReleaseBuffer_F32(conversion_buffer[dnnResourceDst]));
}
