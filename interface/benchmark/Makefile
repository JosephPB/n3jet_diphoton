SRC_DIR=../generic

CXX?=g++
CXXFLAGS=-pedantic -Wall -Wextra -std=c++17 -march=native -mtune=native -O2
CPPFLAGS=-I${SRC_DIR} -I$(NJET_DEV) $(shell pkg-config njet2 eigen3 --cflags)
LDFLAGS=$(shell pkg-config njet2 --libs) -lnjet2tools

.PHONY: clean all

all: time

time: time.o
	${CXX} -o $@ $^ ${LDFLAGS}

time.o: time.cpp ${SRC_DIR}/model_fns.hpp
	${CXX} ${CXXFLAGS} -o $@ -c $< ${CPPFLAGS}

clean:
	-rm *.o time
