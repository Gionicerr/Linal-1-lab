CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -pedantic
TARGET = main.exe
SOURCES = main.cpp tests.cpp

all: $(TARGET)

$(TARGET): $(SOURCES) matrix.h lu.h tests.h
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	del /Q $(TARGET) 2>NUL || exit 0

.PHONY: all run clean
