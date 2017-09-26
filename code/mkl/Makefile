TARGET := conv
WARNINGS := -Wall -Wextra -pedantic -Wno-vla-extension
HEADERS := -isystem /opt/intel/mkl/include
MKL_PATH := /opt/intel/mkl
LIBS := -L $(MKL_PATH)/lib/ -L /usr/local/lib
CXXFLAGS := -std=c++11 -O2

all: conv

conv: $(TARGET).cpp
	$(CXX) $(WARNINGS) $(CXXFLAGS) $(HEADERS) $(LIBS) \
	$(TARGET).cpp -o $(TARGET) \
	-lmkl_intel -lmkl_sequential -lmkl_core \
	-lopencv_imgcodecs -lopencv_imgproc -lopencv_core

.phony: clean

clean:
	rm $(TARGET) mkl-out.png || echo -n ""
