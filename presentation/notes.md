# Notes

Time available: 60 minutes

Basic strategy:
- Inspect a single model (GAN)
- Define an OSI model for neural networks
- Walk down the OSI model only describing (walk it down theoretically)
- Walk it back up, speaking more about code, showing code, comparing libraries (walk it up practically).

Time allocation:

1. Intro, about me etc: 5 minutes
2. How we'll approach the talk (case study, walk up and down osi model). Define OSI model: 5 minutes
3. Walk down: 15-20 minutes
4. Walk up: 20-25 minutes
5. Q & A

## Approach to Talk

I want to show you this field from both a researcher's perspective, who has to
define deep learning models to solve some machine learning task, and an
engineer's perspective, who has to implement the researcher's model in fast
programming languages, on fast hardware.

The approach I want to take is a case study approach. I want to single out a
particular machine learning model that I'm working with and like very much, and
then inspect that model at all levels, from its definition in high level
programming frameworks to the hardware that it ends up running on in production.
And at every level of abstraction I want to explain and show how C++ solves the
particular problems of that subdomain.

To do this, I first want to define an OSI Model for Neural Networks:

6. Task layer (classification, regression, translation, synthesis)
5. Layer layer (basic deep learning building blocks)
4. Graph layer (computation graphs)
3. Op layer (algorithms)
2. Kernel layer (code)
1. Hardware layer

## Down (Theory, Description)

### Task Layer

- What is a GAN

### Layer layer

- Layers of a GAN/dnn - high level mathematical description
- Dense
- Convolutions
- Pooling

### Graph Layer

- Computational graphs, forward backward passes, mapped to Devices
- Distributed computing, downpour sgd

### Op Layer

How are Convolutions implemented (im2col). Fast algorithms for certain operations? -> More research

that softmax paper I have somewhere

This is about __algorithms__, kernel layer is about __implementations__

### Kernel Layer

- Mostly implemented in C++ and CUDA, sometimes in the framework languages
- A kernel typically needs a forward and backward pass
- Discuss code for CPU/GPU implementations of Convolutions
- Importance of GEMM and BLAS.
- Nervana Neon has their own assembly

### Hardware Layer

- Talk a bit about hardware.
- Gpus,
- cpus,
- 200 gpus for alphago
- training time
- data required
- orders of magnitude we're talking about.

## Up (Practice, Code, Libraries)

### Hardware Layer

- Talk about typical hardware setups
- Nvidia GPUs, Big Basin
- CPU side, Intel MKL (and their deep learning library)
- Specialized Hardware, TPUs, Graphcore

### Kernel Layer

- Implement a kernel in tensorflow
- GPU kernel for extra coolness

### Op Layer

- Abstract over them with intel mkl and cudnn

### Graph Layer

- Now introduce deep learning frameworks: tf and caffe2/pytorch
- Explain static vs. dynamic graph
- Explain communication libraries, gloo, mpi

### Layer Layer

- Discuss high level libraries like Keras
- Show TinyDNN or MxNet code to train a GAN

### Task Layer

- Pretrained models
- Show how to load a pretrained tensorflow model in C++
