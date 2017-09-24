TARGET := lenet
WARNINGS := -Wall -Wextra -pedantic
CXXFLAGS := -std=c++11 -O2

all: conv

conv: $(TARGET).cpp
	$(CXX) $(WARNINGS) $(CXXFLAGS) $(HEADERS) $(LIBS) \
	$(TARGET).cpp -o $(TARGET) -ldlib

.phony: clean

clean:
	rm $(TARGET) || echo -n ""
