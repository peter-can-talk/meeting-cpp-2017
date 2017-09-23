if [ ! -d "./MNIST_data" ]; then
  mkdir mnist_data
  cd mnist_data
  wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz \
       http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz \
       http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz \
       http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
  gzip -d *.gz
fi
