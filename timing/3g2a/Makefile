CXX?=g++
CXXFLAGS=-O2 -std=c++17 -Wall -Wextra -pedantic
CPPFLAGS=-I$(NJET_DEV)
CPPFLAGS+=`pkg-config njet2 --cflags`
LDFLAGS=`pkg-config njet2 --libs` -lnjet2tools -lnjet2an0q3g2A

SRCS=$(wildcard *.cpp)
PRGS=$(basename $(SRCS))

all: test

test: test.o
	$(CXX) -o $@ $^ $(LDFLAGS)

test.o: test.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS)

clean:
	-rm *.o $(PRGS)
