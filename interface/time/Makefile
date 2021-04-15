SRC_DIR=../generic

CXX?=g++
CXXFLAGS=-O2 -pedantic -Wall -Wextra -std=c++17
CPPFLAGS=-I${SRC_DIR}
LDFLAGS=

.PHONY: clean all

all: time

time: time.o model_fns.o
	${CXX} -o $@ $^ ${LDFLAGS} -pg

time.o: time.cpp
	${CXX} ${CXXFLAGS} -o $@ -c $< ${CPPFLAGS} -g -pg

model_fns.o: ${SRC_DIR}/model_fns.cpp ${SRC_DIR}/model_fns.hpp
	${CXX} ${CXXFLAGS} -o $@ -c $< ${CPPFLAGS} -g -pg

clean:
	-rm *.o test
