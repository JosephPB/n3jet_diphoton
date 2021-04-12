SRC_DIR=../generic

CXX?=g++
CXXFLAGS=-g -O2 -pedantic -Wall -Wextra -fPIC -DPIC -std=c++17
CPPFLAGS=$(shell pkg-config njet2 --cflags) -I$(shell Sherpa-config --incdir)
LDFLAGS=$(shell pkg-config njet2 --libs) $(shell Sherpa-config --ldflags)
LDFLAGS+=-lqd
DEFS=-DLEGS=5 # choose 5 or 6 for 3g2a or 4g2a
DEFS+=-DRUNS=20 # choose number of NN runs to average over
#DEFS+=-DA=0 # the index of the NN run
# DEFS+=-DUNIT # for generating unit integration grid
# DEFS+=-DNJET # do NJet calculation only (or return NJet result with BOTH)
DEFS+=-DNN # do NN calculation only (or return NN result with BOTH)
# DEFS+=-DBOTH # generate both NN and NJet and choose return by one of two above flags
#DEFS+=-DREC # record results
# DEFS+=-DTIMING # recording runtime information
#DEFS+=-DNAIVE # use naive single NN implementation
DEFS+=-DNN_MODEL="\"./models/RAMBO/events_100k_fks_all_legs_all_pairs_new_sherpa_cuts_pdf_njet/\""

.PHONY: clean all

all: libSherpaNN3g2A.so

interface.o: ${SRC_DIR}/interface.cpp ${SRC_DIR}/interface.hpp Makefile
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS) $(DEFS)

model_fns.o: ${SRC_DIR}/model_fns.cpp ${SRC_DIR}/model_fns.h
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS) $(DEFS)

libSherpaNN3g2A.so: interface.o model_fns.o
	$(CXX) -shared -o $@ $^ $(LDFLAGS) $(LIBS)

clean:
	-rm *.so *.o