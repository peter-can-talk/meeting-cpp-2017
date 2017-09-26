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

tensorflow::Tensor sample_code() {
  using RandomUniform = Eigen::internal::UniformRandomGenerator<float>;

  static std::random_device seed;
  static std::mt19937 rng(seed());
  std::uniform_int_distribution<size_t> indices(0, kDiscreteCodeSize - 1);

  tensorflow::Tensor code(tensorflow::DT_FLOAT,
                          tensorflow::TensorShape({1, kCodeSize}));
  code.flat<float>().setZero();

  const auto one_hot_index = indices(rng);
  code.flat<float>()(one_hot_index) = 1;

  Eigen::array<int, 1> offsets = {{kDiscreteCodeSize}};
  Eigen::array<int, 1> extents = {{kContinuousCodeSize}};
  auto continuous = code.flat<float>().slice(offsets, extents);
  continuous.setRandom<RandomUniform>();
  continuous = (continuous * 2.0f) - 1.0f;

  return code;
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

void save_image(const char* filename, float* buffer) {
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

  auto noise = sample_noise();
  auto code = sample_code();

  std::cout << noise.flat<float>() << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << code.flat<float>() << std::endl;

  tensorflow::Tensor zero(tensorflow::DT_BOOL, tensorflow::TensorShape());
  zero.scalar<bool>()(0) = false;
  std::vector<std::pair<std::string, tensorflow::Tensor>> feeds =
      {{"z:0", noise}, {"c:0", code}, {kLearningPhase, zero}};

  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session->Run(feeds, {"G_final/Tanh"}, {}, &outputs));

  assert(!outputs.empty());
  auto image = outputs.front().flat<float>();
  image = (image + 1.0f) / 2.0f;
  save_image("/tmp/gan-out.png", image.data());

  TF_CHECK_OK(session->Close());
}
