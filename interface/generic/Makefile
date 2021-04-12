CXX?=g++
CXXFLAGS=-g -O2 -pedantic -Wall -Wextra -fPIC -DPIC -std=c++17
CPPFLAGS=$(shell pkg-config njet2 --cflags) -I$(shell Sherpa-config --incdir)
LDFLAGS=$(shell pkg-config njet2 --libs) $(shell Sherpa-config --ldflags)
LDFLAGS+=-lqd
DEFS=-DLEGS=6 # choose 5 or 6 for 3g2a or 4g2a
DEFS+=-DRUNS=20 # choose number of NN runs to average over
DEFS+=-DA=0 # the index of the NN run
# DEFS+=-DUNIT # for generating unit integration grid
# DEFS+=-DNJET # do NJet calculation only (or return NJet result with BOTH)
DEFS+=-DNN # do NN calculation only (or return NN result with BOTH)
# DEFS+=-DBOTH # generate both NN and NJet and choose return by one of two above flags
DEFS+=-DREC # record results
DEFS+=-DTIMING # recording runtime information
DEFS+=-DNAIVE # use naive single NN implementation

.PHONY: clean wipe all

all: interface.so

interface.o: interface.cpp interface.hpp Makefile
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS) $(DEFS)

# NJet4g2A_Interface.o: NN4g2A_Interface.cpp NN4g2A_Interface.hpp Makefile
# 	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS) -DNJET -DREC

model_fns.o: model_fns.cpp model_fns.h
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS) $(DEFS)

# Generate the library file for the Sherpa/NN4g2A interface
interface.so: interface.o model_fns.o
	$(CXX) -shared -o $@ $^ $(LDFLAGS) $(LIBS)

# # Generate the library file for the Sherpa/NJet4g2A interface
# libSherpaNJet4g2A.so: NJet4g2A_Interface.o model_fns.o
# 	$(CXX) -shared -o $@ $^ $(LDFLAGS) $(LIBS)

# Remove compiled files
clean:
	rm -rf *.so *.o OLE_contract_4g2A.lh ex

# Remove everything
wipe: clean
	rm *.pdf
