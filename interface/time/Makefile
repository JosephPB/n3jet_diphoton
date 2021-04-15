SRC_DIR=../generic

CXX?=g++
CXXFLAGS=-O2 -pedantic -Wall -Wextra -std=c++17
CPPFLAGS=-I${SRC_DIR}
LDFLAGS=

.PHONY: clean all

all: time

time: time.o
	${CXX} -o $@ $^ ${LDFLAGS}

time.o: time.cpp ${SRC_DIR}/model_fns.hpp
	${CXX} ${CXXFLAGS} -o $@ -c $< ${CPPFLAGS}

clean:
	-rm *.o test
