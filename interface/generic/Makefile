LOCAL?=$(HOME)/local
NJET_LOCAL?=$(LOCAL)/njet

CXX?=g++

CXXFLAGS=-g -O2 -pedantic -Wall -Wextra -fPIC -DPIC -std=c++17
CPPFLAGS=$(shell pkg-config njet2 --cflags) -I$(shell Sherpa-config --incdir)
# DEFS+=-DDEBUG # turn on verbose debugging output to terminal
# DEFS+=-DTEST # test NN against a single known point
# DEFS+=-DUNIT # for generating unit integration grid
DEFS+=-DBOTH # generate both NN and NJet - if not the only do NN by default or NJet if DNJET on
DEFS+=-DREC # record results
# DEFS+=-DNJET # do NJet calculation and ignore everything else (except PS & res recording)
LDFLAGS=$(shell pkg-config njet2 --libs) $(shell Sherpa-config --ldflags)
LDFLAGS+=-lqd

.PHONY: clean wipe all

all: libSherpaNN4g2A.so libSherpaNJet4g2A.so

NN4g2A_Interface.o: NN4g2A_Interface.cpp NN4g2A_Interface.hpp Makefile
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS) $(DEFS)

NJet4g2A_Interface.o: NN4g2A_Interface.cpp NN4g2A_Interface.hpp Makefile
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS) -DNJET -DREC

model_fns.o: model_fns.cpp model_fns.h Makefile
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS) $(DEFS)

# Generate the library file for the Sherpa/NN4g2A interface
libSherpaNN4g2A.so: NN4g2A_Interface.o model_fns.o
	$(CXX) -shared -o $@ $^ $(LDFLAGS) $(LIBS)

# Generate the library file for the Sherpa/NJet4g2A interface
libSherpaNJet4g2A.so: NJet4g2A_Interface.o model_fns.o
	$(CXX) -shared -o $@ $^ $(LDFLAGS) $(LIBS)

# Remove compiled files
clean:
	rm -rf *.so *.o OLE_contract_4g2A.lh ex

# Remove everything
wipe: clean
	rm *.pdf
