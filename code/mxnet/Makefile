TARGET := lenet
WARNINGS := -Wall -Wextra -pedantic
HEADERS := -isystem $(MXNET_PATH)/cpp-package/include \
					 -isystem $(MXNET_PATH)/include \
 					 -isystem $(MXNET_PATH)/nnvm/include \
					 -isystem $(MXNET_PATH)/dmlc-core/include
LIBS := -L $(MXNET_PATH)/lib
CXXFLAGS := -std=c++11

all: lenet

lenet: $(TARGET).cpp
	$(CXX) $(WARNINGS) $(CXXFLAGS) $(HEADERS) $(LIBS) \
	$(TARGET).cpp -o $(TARGET) -lmxnet

.phony: clean

clean:
	rm $(TARGET) || echo -n ""
