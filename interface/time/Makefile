SRC_DIR=../generic

CXX?=g++
CXXFLAGS=-O3 -pedantic -Wall -Wextra -std=c++17 -march=native -mtune=native
CPPFLAGS=-I${SRC_DIR} $(shell pkg-config eigen3 --cflags)
LDFLAGS=

.PHONY: clean all

all: time

time: time.o
	${CXX} -o $@ $^ ${LDFLAGS}

time.o: time.cpp ${SRC_DIR}/model_fns.hpp
	${CXX} ${CXXFLAGS} -o $@ -c $< ${CPPFLAGS}

clean:
	-rm *.o test
