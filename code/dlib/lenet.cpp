#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <iostream>

namespace lenet {
using namespace dlib;
// clang-format off
  using model = loss_multiclass_log<
      fc<10,
      relu<fc<1024,
      max_pool<2,2,2,2,relu<con<64, 5, 5, 1, 1,
      max_pool<2,2,2,2,relu<con<32, 5, 5, 1, 1,
      input<matrix<uint8_t>>
      >>>>>>>>>>;
// clang-format on
}  // namespace lenet

int main(int argc, char const* argv[]) {
  if (argc < 2) {
    std::cerr << "usage: lenet <mnist_path>" << std::endl;
  }

  std::vector<dlib::matrix<uint8_t>> training_images;
  std::vector<unsigned long> training_labels;

  std::vector<dlib::matrix<uint8_t>> test_images;
  std::vector<unsigned long> test_labels;

  dlib::load_mnist_dataset(argv[1],
                           training_images,
                           training_labels,
                           test_images,
                           test_labels);

  lenet::model model;

  dlib::dnn_trainer<lenet::model> trainer(model);
  trainer.set_learning_rate(0.01);
  trainer.set_min_learning_rate(1e-5);
  trainer.set_mini_batch_size(128);
  trainer.set_max_num_epochs(1);
  trainer.be_verbose();

  trainer.train(training_images, training_labels);

  model.clean();

  std::vector<unsigned long> predicted = model(test_images);
  double hits = 0;
  for (size_t i = 0; i < test_images.size(); i++) {
    if (predicted[i] == test_labels[i]) {
      hits += 1;
    }
  }

  std::cerr << "Test accuracy: " << hits / test_images.size() << std::endl;
}
