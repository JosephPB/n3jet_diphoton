SRC_DIR=../generic

CXX?=g++
CXXFLAGS=-O2 -pedantic -Wall -Wextra -std=c++17 -g
CPPFLAGS=-I${SRC_DIR}
LDFLAGS=

.PHONY: clean all

all: test

test: test.o model_fns.o
	${CXX} -o $@ $^ ${LDFLAGS}

model_fns.o: ${SRC_DIR}/model_fns.cpp ${SRC_DIR}/model_fns.hpp
	${CXX} ${CXXFLAGS} -o $@ -c $< ${CPPFLAGS}

clean:
	-rm *.o test
