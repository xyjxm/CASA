LATEXMK ?= latexmk
PYTHON ?= python3

.PHONY: all figures tables check clean

all: check
	$(LATEXMK) -pdf -interaction=nonstopmode main.tex

figures tables:
	$(PYTHON) scripts/generate_paper_artifacts.py

check: figures tables
	@$(PYTHON) -c "import json; assert json.load(open('table_consistency_check.json'))['status'] == 'pass'"

clean:
	$(LATEXMK) -C main.tex
	rm -f main.bbl main.blg main.fls main.fdb_latexmk main.synctex.gz
