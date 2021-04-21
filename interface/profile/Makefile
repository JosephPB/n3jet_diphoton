MDL_DIR=../generic
TME_DIR=../time

CXX?=g++
CXXFLAGS=-Og -pedantic -Wall -Wextra -std=c++17 -march=native -mtune=native
CPPFLAGS=-I${MDL_DIR} $(shell pkg-config eigen3 --cflags)
LDFLAGS=

PRF_CPL=-g -pg
PRF_LNK=-pg

.PHONY: clean all

all: time

time: time.o
	${CXX} -o $@ $^ ${LDFLAGS} ${PRF_LNK}

time.o: ${TME_DIR}/time.cpp ${MDL_DIR}/model_fns.hpp
	${CXX} ${CXXFLAGS} -o $@ -c $< ${CPPFLAGS} ${PRF_CPL}

clean:
	-rm *.o test
