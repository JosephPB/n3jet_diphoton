SRC_DIR=../generic

CXX?=g++
CXXFLAGS=-g -O2 -pedantic -Wall -Wextra -fPIC -DPIC -std=c++14 -march=native -mtune=native
CPPFLAGS=$(shell pkg-config njet2 eigen3 --cflags) -I$(shell Sherpa-config --incdir)
LDFLAGS=$(shell pkg-config njet2 --libs) $(shell Sherpa-config --ldflags)
LDFLAGS+=-lqd
DEFS=-DLEGS=6 # choose 5 or 6 for 3g2a or 4g2a
DEFS+=-DRUNS=20 # choose number of NN runs to average over
#DEFS+=-DA=0 # the index of the NN run
# DEFS+=-DUNIT # for generating unit integration grid
# DEFS+=-DNJET # do NJet calculation only (or return NJet result with BOTH)
DEFS+=-DNN # do NN calculation only (or return NN result with BOTH)
# DEFS+=-DBOTH # generate both NN and NJet and choose return by one of two above flags
#DEFS+=-DREC # record results
# DEFS+=-DTIMING # recording runtime information
#DEFS+=-DNAIVE # use naive single NN implementation
DEFS+=-DNN_MODEL="\"./models/4g2A/RAMBO/100k_unit_0001_fks/\""

.PHONY: clean all

all: libSherpaNN4g2A.so

interface.o: ${SRC_DIR}/interface.cpp ${SRC_DIR}/interface.hpp ${SRC_DIR}/model_fns.hpp
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS) $(DEFS)

libSherpaNN4g2A.so: interface.o model_fns.o
	$(CXX) -shared -o $@ $^ $(LDFLAGS) $(LIBS)

clean:
	-rm *.so *.o
