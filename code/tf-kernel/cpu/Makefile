TARGET := kernel
HEADERS := -I `python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'`
CXXFLAGS := -std=c++11 -fPIC -O2

ifeq ($(shell uname), Darwin)
	CXXFLAGS := $(CXXFLAGS) -undefined dynamic_lookup
endif

all: cpu

cpu: $(TARGET).cpp
	$(CXX) $(HEADERS) $(CXXFLAGS) $(TARGET).cpp -shared -o $(TARGET).so

.phony: clean

clean:
	rm $(TARGET).so || echo -n ""
