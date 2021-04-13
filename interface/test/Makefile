SRC_DIR=../generic

CXX?=g++
CXXFLAGS=-g -O2 -pedantic -Wall -Wextra -std=c++17
CPPFLAGS=-I${SRC_DIR}
LDFLAGS=

.PHONY: clean all

test: test.o model_fns.o
	${CXX} -o $@ $^ ${LDFLAGS}

model_fns.o: ${SRC_DIR}/model_fns.cpp ${SRC_DIR}/model_fns.h
	${CXX} ${CXXFLAGS} -o $@ -c $< ${CPPFLAGS}

clean:
	-rm *.o test
