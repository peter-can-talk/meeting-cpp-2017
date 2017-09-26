TARGET := lenet
WARNINGS := -Wall -Wextra -pedantic
HEADERS := -isystem $(MXNET_PATH)/cpp-package/include \
					 -isystem $(MXNET_PATH)/include \
 					 -isystem $(MXNET_PATH)/nnvm/include \
					 -isystem $(MXNET_PATH)/dmlc-core/include \
					 -I ../
LIBS := -L $(MXNET_PATH)/lib -L /usr/local/lib
CXXFLAGS := -std=c++11

all: lenet

lenet: $(TARGET).cpp
	$(CXX) $(WARNINGS) $(CXXFLAGS) $(HEADERS) $(LIBS) \
	$(TARGET).cpp ../common/socket.cpp -o $(TARGET) \
	-lmxnet -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

.phony: clean

clean:
	rm $(TARGET) || echo -n ""
