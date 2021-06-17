SRC_DIR=../../interface/generic
COM_DIR=../common

CXX?=g++
CXXFLAGS=-O2 -std=c++14 -Wall -Wextra -pedantic -march=native -mtune=native
CPPFLAGS=`pkg-config njet3 eigen3 --cflags` -I${SRC_DIR} -I${COM_DIR}
LDFLAGS=`pkg-config njet3 --libs` -lnjet3tools

SRCS=$(wildcard *.cpp)
PRGS=$(basename $(SRCS))

all: test

test: test.o ${COM_DIR}/timing.o
	$(CXX) -o $@ $^ $(LDFLAGS)

test.o: test.cpp ${SRC_DIR}/model_fns.hpp
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS)

clean:
	-rm *.o $(PRGS)
