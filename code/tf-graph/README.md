# tf-graph

Demo of loading and running a TensorFlow graph trained and exported in Python.

## Building

### Prerequisites

1. A clone of TensorFlow: `git clone https://github.com/tensorflow/tensorflow` checked out at release/branch r1.3 (that's what I use, may work with newer),
2. See `https://github.com/tensorflow/tensorflow/issues/2412` how to build a library from TensorFlow sources (you'll need Bazel),
3. A clone of Protocol Buffers at version 3.3.0 (exactly!): `https://github.com/google/protobuf/releases/tag/v3.3.0`,
4. Build that version of ProtoBuf from source using the Makefile inside,
5. OpenCV2 (consult your package manager).

Note that TensorFlow also needs Eigen.

### Build the Graph Loader

You can build using the Makefile found in this folder. You need to set `TF_PATH` and `PB_PATH` environment variables to your local folder location of TensorFlow and ProtoBuf, respectively, e.g. for me:

```sh
TF_PATH=~/Documents/Libraries/tensorflow \
PB_PATH=~/Documents/Libraries/protobuf-3.3.0 make
```

This will build `load-graph.cpp`, which loads a graph and generates an image.
There is also `load-graph-server.cpp` under the `server` target of the Makefile
which builds the version that listens on a socket for inference requests from
the demo.

### Demo

First make sure you've built the server version with the above instructions.
Then, for the `demo` folder, you'll need Qt (5.7 or newer). Generate a Makefile
and simply make:

```sh
$ cd demo
$ qmake
$ make
```

## Running

Both the server and non-server version require two arguments:

1. The path to a model checkpoint,
2. The path prefix for a saved TensorFlow session.

You can pass those two to the binary of `load-graph.cpp` and it will generate an
image under `/tmp/out.png`.

For the demo, the server version of `load-graph` will also start listening on a
socket when you run it. You should start this binary first, then start the
`demo` Qt app, which will connect to the server to request images. That is:

```sh
$ LD_LIBRARY_PATH=/path/to/tensorflow_cc.so load-graph-server graph.pb checkpoint
$ demo/<demo binary>
```

where `<demo binary>` is `demo.app/Contents/MacOS/demo` for example. Differs on
Linux or Windows.
