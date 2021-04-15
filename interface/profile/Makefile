MDL_DIR=../generic
TME_DIR=../time

CXX?=g++
CXXFLAGS=-O2 -pedantic -Wall -Wextra -std=c++17
CPPFLAGS=-I${MDL_DIR}
LDFLAGS=

PRF_CPL=-g -pg
PRF_LNK=-pg

.PHONY: clean all

all: time

time: time.o model_fns.o
	${CXX} -o $@ $^ ${LDFLAGS} ${PRF_LNK}

time.o: ${TME_DIR}/time.cpp
	${CXX} ${CXXFLAGS} -o $@ -c $< ${CPPFLAGS} ${PRF_CPL}

model_fns.o: ${MDL_DIR}/model_fns.cpp ${MDL_DIR}/model_fns.hpp
	${CXX} ${CXXFLAGS} -o $@ -c $< ${CPPFLAGS} ${PRF_CPL}

clean:
	-rm *.o test
