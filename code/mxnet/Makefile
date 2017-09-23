TARGET := lenet
HEADERS := -I $(MXNET_PATH)/cpp-package/include -I $(MXNET_PATH)/include -I $(MXNET_PATH)/nnvm/include -I $(MXNET_PATH)/dmlc-core/include
LIBS := -L $(MXNET_PATH)/lib
CXXFLAGS := -std=c++11

all: lenet

lenet: $(TARGET).cpp
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(TARGET).cpp -o $(TARGET) -lmxnet

.phony: clean

clean:
	rm $(TARGET).cpp || echo -n ""
