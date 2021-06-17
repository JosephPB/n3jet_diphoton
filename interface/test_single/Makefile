SRC_DIR=../generic

CXX?=g++
CXXFLAGS=-O2 -pedantic -Wall -Wextra -std=c++14 -march=native -mtune=native
CPPFLAGS=-I${SRC_DIR} $(shell pkg-config eigen3 --cflags)
LDFLAGS=

.PHONY: clean all

all: test

test: test.o
	${CXX} -o $@ $^ ${LDFLAGS}

test.o: test.cpp ${SRC_DIR}/model_fns.hpp
	${CXX} ${CXXFLAGS} -o $@ -c $< ${CPPFLAGS}

clean:
	-rm *.o test
