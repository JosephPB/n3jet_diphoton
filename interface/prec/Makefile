SRC_DIR=../generic

CXX?=g++
CXXFLAGS=-pedantic -Wall -Wextra -std=c++17 -march=native -mtune=native -O2
CPPFLAGS=-I${SRC_DIR} $(shell pkg-config njet3 eigen3 --cflags)
LDFLAGS=$(shell pkg-config njet3 --libs) -lnjet3tools

.PHONY: clean all

all: prec

prec: prec.o
	${CXX} -o $@ $^ ${LDFLAGS}

prec.o: prec.cpp ${SRC_DIR}/model_fns.hpp
	${CXX} ${CXXFLAGS} -O2 -o $@ -c $< ${CPPFLAGS}

clean:
	-rm *.o prec
