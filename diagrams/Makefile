.PHONY: all clean wipe

all: main.pdf

main.pdf: main.tex
	mkdir -p graphics
	pdflatex -shell-escape $<

clean:
	-rm *.aux *.auxlock *.log */*.log */*.dpth */*.md5

wipe: clean
	-rm -r graphics/ *.pdf
