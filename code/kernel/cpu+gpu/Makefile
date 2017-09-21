TARGET := kernel
HEADERS := -I `python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'`
CXXFLAGS := -std=c++11 -fPIC -O2 -D GOOGLE_CUDA=1
GPUFLAGS := -arch=sm_35 -std=c++11 -O2 -D GOOGLE_CUDA=1 \
						-x cu -Xcompiler -fPIC --expt-relaxed-constexpr

ifeq ($(shell uname), Darwin)
	CXXFLAGS := $(CXXFLAGS) -undefined dynamic_lookup
endif

all: cpu+gpu

gpu: $(TARGET).cu
	nvcc $(HEADERS) $(GPUFLAGS) -c $(TARGET).cu -o $(TARGET).cu.o

cpu: $(TARGET).cpp
	$(CXX) $(HEADERS) $(CXXFLAGS) -c $(TARGET).cpp -o $(TARGET).o

cpu+gpu: cpu gpu
	$(CXX) $(HEADERS) $(CXXFLAGS) -L /usr/local/cuda/lib64 \
	$(TARGET).o $(TARGET).cu.o -shared -o $(TARGET).so -fPIC -lcudart

.phony: clean

clean:
	rm $(TARGET).so $(TARGET).o $(TARGET).cu.o || echo -n ""
