.PHONY: all experiments paper clean

# Use TinyTeX binaries if available
TINYTEX := $(HOME)/Library/TinyTeX/bin/universal-darwin
XELATEX := $(shell which xelatex || echo xelatex)
BIBTEX := $(shell [ -x "$(TINYTEX)/bibtex" ] && echo "$(TINYTEX)/bibtex" || echo bibtex)

all: experiments paper

experiments:
	cd experiments && python3 run_experiments.py

paper: experiments
	$(XELATEX) -interaction=nonstopmode rank-preserving-multiclass.tex
	$(BIBTEX) rank-preserving-multiclass
	$(XELATEX) -interaction=nonstopmode rank-preserving-multiclass.tex
	$(XELATEX) -interaction=nonstopmode rank-preserving-multiclass.tex

draft:
	$(XELATEX) -interaction=nonstopmode rank-preserving-multiclass.tex
	$(BIBTEX) rank-preserving-multiclass
	$(XELATEX) -interaction=nonstopmode rank-preserving-multiclass.tex
	$(XELATEX) -interaction=nonstopmode rank-preserving-multiclass.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	rm -rf experiments/output
