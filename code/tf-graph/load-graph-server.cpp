#include "common/socket.h"

#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/errors.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/util/command_line_flags.h>

#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

const size_t kNoiseSize = 100;
const size_t kDiscreteCodeSize = 10;
const size_t kContinuousCodeSize = 2;
const size_t kCodeSize = kDiscreteCodeSize + kContinuousCodeSize;
const char* const kLearningPhase = "batch_normalization_1/keras_learning_phase";

void load_graph(const std::string& graph_path,
                std::unique_ptr<tensorflow::Session>& session) {
  tensorflow::GraphDef graph_def;
  TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                          graph_path,
                                          &graph_def));

  session.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_CHECK_OK(session->Create(graph_def));
}

tensorflow::Tensor sample_noise() {
  using RandomNormal = Eigen::internal::NormalRandomGenerator<float>;
  tensorflow::Tensor noise(tensorflow::DT_FLOAT,
                           tensorflow::TensorShape({1, kNoiseSize}));
  noise.matrix<float>().setRandom<RandomNormal>();
  return noise;
}

tensorflow::Tensor create_code(int digit, double a, double b) {
  tensorflow::Tensor tensor(tensorflow::DT_FLOAT,
                            tensorflow::TensorShape({1, kCodeSize}));
  auto code = tensor.flat<float>();
  code.setZero();

  assert(digit >= 0 && digit <= 9);
  code(digit) = 1;
  code(kDiscreteCodeSize) = a;
  code(kDiscreteCodeSize + 1) = b;

  return tensor;
}

tensorflow::Tensor generate(std::unique_ptr<tensorflow::Session>& session,
                            tensorflow::Tensor& noise,
                            tensorflow::Tensor& code) {
  tensorflow::Tensor zero(tensorflow::DT_BOOL, tensorflow::TensorShape());
  zero.scalar<bool>()(0) = false;
  std::vector<std::pair<std::string, tensorflow::Tensor>> feeds =
      {{"z:0", noise}, {"c:0", code}, {kLearningPhase, zero}};

  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session->Run(feeds, {"G_final/Tanh:0"}, {}, &outputs));

  assert(!outputs.empty());
  return outputs.front();
}

void restore_session(const std::string& checkpoint_path,
                     std::unique_ptr<tensorflow::Session>& session) {
  tensorflow::Tensor checkpoint_tensor(tensorflow::DT_STRING,
                                       tensorflow::TensorShape());
  checkpoint_tensor.flat<tensorflow::string>()(0) = checkpoint_path;
  TF_CHECK_OK(session->Run({{"save/Const:0", checkpoint_tensor}},
                           {},
                           {"save/restore_all"},
                           nullptr));
  LOG(INFO) << "Restored session from " << checkpoint_path;
}

void save_image(const std::string& filename, float* buffer) {
  cv::Mat image(28, 28, CV_32F, buffer);
  cv::normalize(image, image, 0.0, 255.0, cv::NORM_MINMAX);
  image.convertTo(image, CV_8UC3);
  cv::imwrite(filename, image);
  LOG(INFO) << "Wrote " << filename;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: load-graph <path/to/graph> <checkpoint>\n";
    std::exit(EXIT_FAILURE);
  }

  tensorflow::port::InitMain(argv[0], &argc, &argv);

  std::unique_ptr<tensorflow::Session> session;
  load_graph(argv[1], session);
  restore_session(argv[2], session);

  Socket socket(6666);
  std::cerr << "Listening on port 6666" << std::endl;

  socket.accept();
  std::cerr << "Connection established" << std::endl;

  for (size_t count = 0; true; ++count) {
    std::istringstream stream(socket.read(256));
    int digit = 0;
    double a = 0, b = 0;
    stream >> digit >> a >> b;

    LOG(INFO) << "Prediction request for code: \"" << digit << " " << a << " "
              << b << "\"";

    auto noise = sample_noise();
    auto code = create_code(digit, a, b);
    auto tensor = generate(session, noise, code);
    auto image = tensor.flat<float>();
    image = (image + 1.0f) / 2.0f;
    const std::string image_path =
        "/tmp/gan-out-" + std::to_string(count % 2) + ".png";
    save_image(image_path, image.data());

    socket.write(image_path);

    LOG(INFO) << "Wrote " << image_path;
  }

  TF_CHECK_OK(session->Close());
}
