CXX := nvcc
TARGET := conv
CXXFLAGS := -arch=sm_35 -std=c++11 -O2

all: conv

conv: $(TARGET).cu
	$(CXX) $(CXXFLAGS) $(TARGET).cu -o $(TARGET)

.phony: clean

clean:
	rm $(TARGET) || echo -n ""
