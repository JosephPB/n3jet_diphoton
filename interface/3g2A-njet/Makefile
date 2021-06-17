SRC_DIR=../generic

CXX?=g++
CXXFLAGS=-O2 -pedantic -Wall -Wextra -fPIC -DPIC -std=c++14 -march=native -mtune=native
CPPFLAGS=$(shell pkg-config njet3 --cflags) -I$(shell Sherpa-config --incdir)
LDFLAGS=$(shell pkg-config njet3 --libs) $(shell Sherpa-config --ldflags)
DEFS=-DLEGS=5 # choose 5 or 6 for 3g2a or 4g2a
# DEFS+=-DDELTA=0.02 # choose delta (y_cut from the paper)
# DEFS+=-DRUNS=20 # choose number of NN runs to average over
# DEFS+=-DA=0 # the index of the NN run
# DEFS+=-DUNIT # for generating unit integration grid
DEFS+=-DNJET # do NJet calculation only (or return NJet result with BOTH)
# DEFS+=-DNN # do NN calculation only (or return NN result with BOTH)
# DEFS+=-DBOTH # generate both NN and NJet and choose return by one of two above flags
DEFS+=-DREC # record results
# DEFS+=-DTIMING # recording runtime information
# DEFS+=-DNAIVE # use naive single NN implementation
# DEFS+=-DNN_MODEL="\"./models/RAMBO/events_100k_fks_all_legs_all_pairs_new_sherpa_cuts_pdf_njet/\""

.PHONY: clean all

all: libSherpaNJet3g2A.so OLE_contract_3g2A.lh

interface.o: ${SRC_DIR}/interface.cpp ${SRC_DIR}/interface.hpp
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS) $(DEFS)

libSherpaNJet3g2A.so: interface.o
	$(CXX) -shared -o $@ $^ $(LDFLAGS) $(LIBS)

OLE_contract_3g2A.lh: OLE_order_3g2A.lh
	njet.py -o $@ $<

clean:
	-rm *.so *.o OLE_contract_3g2A.lh
