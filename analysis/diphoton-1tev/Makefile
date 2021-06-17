.PHONY: all clean

all: Rivetdiphoton.so

Rivetdiphoton.so: diphoton.cpp
	rivet-buildplugin -j $(shell nproc) $@ $<

clean:
	-rm Rivetdiphoton.so
