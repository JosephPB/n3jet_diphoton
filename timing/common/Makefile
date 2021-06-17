CXX?=g++
CXXFLAGS=-O2 -std=c++17 -Wall -Wextra -pedantic -march=native -mtune=native

all: timing.o

timing.o: timing.cpp timing.hpp
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS)

clean:
	-rm *.o
