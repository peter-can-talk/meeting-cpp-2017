TARGET := load-graph
WARNINGS := -Wall -Wextra -pedantic -Wno-vla-extension
HEADERS := -isystem $(TF_PATH) \
					 -isystem $(TF_PATH)/bazel-genfiles \
					 -isystem $(TF_PATH)/third-party \
					 -isystem /usr/local/include/eigen3 \
					 -isystem $(PB_PATH)/src
LIBS := -L$(TF_PATH)/bazel-bin/tensorflow/
CXXFLAGS := -std=c++11 -O2

all: graph

graph: $(TARGET).cpp
	$(CXX) $(WARNINGS) $(CXXFLAGS) $(HEADERS) $(LIBS) \
	$(TARGET).cpp -o $(TARGET) -l tensorflow_cc \
	-lopencv_imgcodecs -lopencv_imgproc -lopencv_core

server: $(TARGET)-server.cpp
	$(CXX) $(WARNINGS) $(CXXFLAGS) $(HEADERS) $(LIBS) \
	$(TARGET)-server.cpp socket.cpp -o $(TARGET)-server -l tensorflow_cc \
	-lopencv_imgcodecs -lopencv_imgproc -lopencv_core

.phony: clean

clean:
	rm $(TARGET) $(TARGET)-server || echo -n ""
