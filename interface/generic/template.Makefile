CXX?=g++
CXXFLAGS=-g -O2 -pedantic -Wall -Wextra -fPIC -DPIC -std=c++17
CPPFLAGS=$(shell pkg-config njet2 --cflags) -I$(shell Sherpa-config --incdir)
LDFLAGS=$(shell pkg-config njet2 --libs) $(shell Sherpa-config --ldflags)
LDFLAGS+=-lqd
DEFS=-DLEGS=5 # choose 5 or 6 for 3g2a or 4g2a
DEFS+=-DRUNS=20 # choose number of NN runs to average over
DEFS+=-DA=AAAAA # the index of the NN run
# DEFS+=-DUNIT # for generating unit integration grid
# DEFS+=-DNJET # do NJet calculation only (or return NJet result with BOTH)
DEFS+=-DNN # do NN calculation only (or return NN result with BOTH)
# DEFS+=-DBOTH # generate both NN and NJet and choose return by one of two above flags
DEFS+=-DREC # record results
DEFS+=-DTIMING # recording runtime information

.PHONY: clean all

all: interface-AAAAA.so

interface.o: interface.cpp interface.hpp Makefile
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS) $(DEFS)

model_fns.o: model_fns.cpp model_fns.h
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS) $(DEFS)

interface-AAAAA.so: interface.o model_fns.o
	$(CXX) -shared -o $@ $^ $(LDFLAGS) $(LIBS)

clean:
	rm -rf *.so *.o *.lh
