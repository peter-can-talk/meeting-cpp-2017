#include <mxnet-cpp/MxNetCpp.h>

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>

namespace mx = mxnet::cpp;

mx::Symbol LeNet() {
  auto images = mx::Symbol::Variable("images");
  auto labels = mx::Symbol::Variable("labels");

  // ------------------------------- CONV 1 -------------------------------

  mx::Symbol conv1_weights("conv1_weights");
  mx::Symbol conv1_bias("conv1_bias");

  auto conv1 = mx::Convolution("conv1",
                               images,
                               conv1_weights,
                               conv1_bias,
                               /*kernel=*/mx::Shape(5, 5),
                               /*filters=*/32);
  auto conv1_activation =
      mx::Activation("conv1_activation", conv1, mx::ActivationActType::kRelu);
  auto pool1 = mx::Pooling("pool1",
                           conv1_activation,
                           mx::Shape(2, 2),
                           mx::PoolingPoolType::kMax,
                           /*global_pool=*/false,
                           /*use_cudnn=*/false,
                           mx::PoolingPoolingConvention::kValid,
                           mx::Shape(2, 2));

  // ------------------------------- CONV 2 -------------------------------

  mx::Symbol conv2_weights("conv2_weights");
  mx::Symbol conv2_bias("conv2_bias");

  auto conv2 = mx::Convolution("conv2",
                               pool1,
                               conv2_weights,
                               conv2_bias,
                               /*kernel=*/mx::Shape(5, 5),
                               /*filters=*/64);
  auto conv2_activation =
      mx::Activation("conv2_activation", conv2, mx::ActivationActType::kRelu);
  auto pool2 = mx::Pooling("pool2",
                           conv2_activation,
                           mx::Shape(2, 2),
                           mx::PoolingPoolType::kMax,
                           /*global_pool=*/false,
                           /*use_cudnn=*/false,
                           mx::PoolingPoolingConvention::kValid,
                           mx::Shape(2, 2));

  // ------------------------------- FC 1 -------------------------------

  mx::Symbol fc1_weights("fc1_weights");
  mx::Symbol fc1_bias("fc1_bias");

  auto flatten = mx::Flatten("flatten", pool2);
  auto fc1 = mx::FullyConnected("fc1",
                                flatten,
                                fc1_weights,
                                fc1_bias,
                                /*units=*/1024);
  auto fc1_activation =
      mx::Activation("fc1_activation", fc1, mx::ActivationActType::kRelu);

  // ------------------------------- FC 2 -------------------------------

  mx::Symbol fc2_weights("fc2_weights");
  mx::Symbol fc2_bias("fc2_bias");

  auto fc2 = mx::FullyConnected("fc2",
                                fc1_activation,
                                fc2_weights,
                                fc2_bias,
                                /*units=*/10);

  // ------------------------------- P -------------------------------

  return mx::SoftmaxOutput("softmax", fc2, labels);
}

int main(int argc, char const* argv[]) {
  if (argc < 2) {
    std::cerr << "usage: lenet <mnist_path>\n";
    std::exit(EXIT_FAILURE);
  }

  std::string mnist_path(argv[1]);

  const size_t batch_size = 128;
  const size_t number_of_epochs = 100;
  const size_t image_width = 28;
  const size_t image_height = 28;
  const size_t image_channels = 1;

  auto context = mx::Context::cpu();

  mx::Shape image_shape(batch_size, image_channels, image_width, image_height);

  auto lenet = LeNet();

  // clang-format off
  std::map<std::string, mx::NDArray> symbols = {
    {"images", mx::NDArray(image_shape, context)},
    {"labels", mx::NDArray(mx::Shape(batch_size), context)},
  };
  // clang-format on

  lenet.InferArgsMap(context, &symbols, symbols);
  const auto symbol_names = lenet.ListArguments();

  mx::Normal normal_initializer(/*mean=*/0.0, /*stddev=*/0.1);
  for (auto& symbol : symbols) {
    if (symbol.first == "images" || symbol.first == "labels") continue;
    normal_initializer(symbol.first, &symbol.second);
  }

  mx::Optimizer* optimizer = mx::OptimizerRegistry::Find("sgd");
  assert(optimizer != nullptr);
  optimizer->SetParam("lr", 0.1)->SetParam("rescale_grad", 1.0 / batch_size);

  std::unique_ptr<mx::Executor> executor(lenet.SimpleBind(context, symbols));

  auto training_iterator =
      mx::MXDataIter("MNISTIter")
          .SetParam("image", mnist_path + "/train-images-idx3-ubyte")
          .SetParam("label", mnist_path + "/train-labels-idx1-ubyte")
          .SetParam("batch_size", batch_size)
          .SetParam("shuffle", true)
          .SetParam("flat", 0)
          .CreateDataIter();

  auto test_iterator =
      mx::MXDataIter("MNISTIter")
          .SetParam("image", mnist_path + "/t10k-images-idx3-ubyte")
          .SetParam("label", mnist_path + "/t10k-labels-idx1-ubyte")
          .SetParam("batch_size", batch_size)
          .SetParam("shuffle", true)
          .SetParam("flat", 0)
          .CreateDataIter();

  size_t training_number_of_batches = 60000 / batch_size;
  for (size_t epoch = 1; epoch <= number_of_epochs; ++epoch) {
    training_iterator.Reset();
    for (size_t batch_index = 0; training_iterator.Next(); ++batch_index) {
      auto batch = training_iterator.GetDataBatch();
      batch.data.CopyTo(&symbols["images"]);
      batch.label.CopyTo(&symbols["labels"]);

      // Wait for symbols to be populated.
      mx::NDArray::WaitAll();

      executor->Forward(/*training=*/true);
      executor->Backward();

      for (size_t symbol = 0; symbol < symbol_names.size(); ++symbol) {
        if (symbol_names[symbol] == "images") continue;
        if (symbol_names[symbol] == "labels") continue;
        optimizer->Update(symbol,
                          executor->arg_arrays[symbol],
                          executor->grad_arrays[symbol]);
      }

      std::cout << "\rBatch " << batch_index << "/"
                << training_number_of_batches << std::flush;
    }

    std::cout << std::endl;
    LOG(INFO) << "Evaluating ...";

    mx::Accuracy accuracy;
    test_iterator.Reset();
    while (test_iterator.Next()) {
      auto batch = test_iterator.GetDataBatch();
      batch.data.CopyTo(&symbols["images"]);
      batch.label.CopyTo(&symbols["labels"]);
      mx::NDArray::WaitAll();
      executor->Forward(/*training=*/false);
      accuracy.Update(batch.label, executor->outputs[0]);
    }

    std::cout << "Epoch: " << epoch << " | Accuracy: " << accuracy.Get()
              << std::endl;
  }

  MXNotifyShutdown();
}
