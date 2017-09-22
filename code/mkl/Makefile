TARGET := conv
HEADERS := -I /opt/intel/mkl/include
LIBS := -L /opt/intel/mkl/lib/ -L/usr/local/lib
CXXFLAGS := -std=c++11 -g

all: conv

conv: $(TARGET).cpp
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(TARGET).cpp -o $(TARGET) \
	-lmkl_intel -lmkl_sequential -lmkl_core \
	-lopencv_imgcodecs -lopencv_imgproc -lopencv_core

.phony: clean

clean:
	rm $(TARGET).cpp || echo -n ""
