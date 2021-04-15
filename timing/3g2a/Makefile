SRC_DIR=../../interface/generic

CXX?=g++
CXXFLAGS=-O2 -std=c++17 -Wall -Wextra -pedantic
CPPFLAGS=`pkg-config njet2 --cflags` -I$(NJET_DEV) -I${SRC_DIR}
LDFLAGS=`pkg-config njet2 --libs` -lnjet2tools -lnjet2an0q3g2A

SRCS=$(wildcard *.cpp)
PRGS=$(basename $(SRCS))

all: test

test: test.o model_fns.o
	$(CXX) -o $@ $^ $(LDFLAGS)

test.o: test.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS)

model_fns.o: ${SRC_DIR}/model_fns.cpp ${SRC_DIR}/model_fns.hpp
	${CXX} ${CXXFLAGS} -o $@ -c $< ${CPPFLAGS}

clean:
	-rm *.o $(PRGS)
