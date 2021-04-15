.PHONY: all clean

all: Rivetdiphoton.so

Rivetdiphoton.so: diphoton.cc
	rivet-buildplugin -j $(shell nproc) $@ $<

clean:
	-rm Rivetdiphoton.so
