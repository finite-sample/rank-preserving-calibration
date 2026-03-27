.PHONY: all experiments paper clean

# Use TinyTeX binaries if available
TINYTEX := $(HOME)/Library/TinyTeX/bin/universal-darwin
XELATEX := $(shell which xelatex || echo xelatex)
BIBTEX := $(shell [ -x "$(TINYTEX)/bibtex" ] && echo "$(TINYTEX)/bibtex" || echo bibtex)

all: experiments paper

experiments:
	cd experiments && python3 run_experiments.py

paper: experiments
	$(XELATEX) -interaction=nonstopmode ms.tex
	$(BIBTEX) ms
	$(XELATEX) -interaction=nonstopmode ms.tex
	$(XELATEX) -interaction=nonstopmode ms.tex

draft:
	$(XELATEX) -interaction=nonstopmode ms.tex
	$(BIBTEX) ms
	$(XELATEX) -interaction=nonstopmode ms.tex
	$(XELATEX) -interaction=nonstopmode ms.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	rm -rf experiments/output
