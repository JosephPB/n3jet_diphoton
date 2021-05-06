SRC_DIR=../../interface/generic
COM_DIR=../common

CXX?=g++
CXXFLAGS=-O2 -std=c++17 -Wall -Wextra -pedantic
CPPFLAGS=`pkg-config njet2 eigen3 --cflags` -I$(NJET_DEV) -I${SRC_DIR} -I${COM_DIR}
LDFLAGS=`pkg-config njet2 --libs` -lnjet2tools
LDFLAGS+=-lnjet2an0q2g2A
LDFLAGS+=-lnjet2an0q3g2A
LDFLAGS+=-lnjet2an0q4g
LDFLAGS+=-lnjet2an0q5g 
# LDFLAGS+=-lnjet2an2q3g

SRCS=$(wildcard *.cpp)
PRGS=$(basename $(SRCS))

all: test

test: test.o ${COM_DIR}/timing.o
	$(CXX) -o $@ $^ $(LDFLAGS)

test.o: test.cpp ${SRC_DIR}/model_fns.hpp
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS)

clean:
	-rm *.o $(PRGS)
