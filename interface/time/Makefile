SRC_DIR=../generic

CXX?=g++
CXXFLAGS=-pedantic -Wall -Wextra -std=c++17 -march=native -mtune=native
CPPFLAGS=-I${SRC_DIR} $(shell pkg-config eigen3 --cflags)
LDFLAGS=

.PHONY: clean all

all: time_O2 time_O3

time_O2: time_O2.o
	${CXX} -o $@ $^ ${LDFLAGS}

time_O2.o: time.cpp ${SRC_DIR}/model_fns.hpp
	${CXX} ${CXXFLAGS} -O2 -o $@ -c $< ${CPPFLAGS}

time_O3: time_O3.o
	${CXX} -o $@ $^ ${LDFLAGS}

time_O3.o: time.cpp ${SRC_DIR}/model_fns.hpp
	${CXX} ${CXXFLAGS} -O2 -o $@ -c $< ${CPPFLAGS}

clean:
	-rm *.o test
