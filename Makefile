# PRAMAN — one-command reproduction. CPU-only, offline runtime.
# Usage: make setup && make data && make slice    (then make full for final numbers)
PY ?= .venv/bin/python
RUN ?= slice

.PHONY: setup data validate-crc score pipeline slice full ood test report clean all

setup:            ## create venv + install CPU deps + lockfile
	bash scripts/00_setup.sh

data:             ## download RAGTruth + build claim cache
	$(PY) scripts/10_data.py

validate-crc:     ## prove the conformal risk control math on a synthetic toy
	$(PY) scripts/31_crc_validate.py

score:            ## run the verifier, cache scores (RUN=slice|full)
	$(PY) scripts/20_score.py --run-id $(RUN)

pipeline:         ## calibrate + CRC + validate the guarantee on cached scores
	$(PY) scripts/30_pipeline.py --run-id $(RUN)

slice:            ## fast end-to-end on a small subset
	$(PY) scripts/20_score.py --run-id slice --max-train 3000 --max-test 1500
	$(PY) scripts/30_pipeline.py --run-id slice

full:             ## full-size run for the headline numbers
	$(PY) scripts/20_score.py --run-id full --max-train 100000 --max-test 100000
	$(PY) scripts/30_pipeline.py --run-id full

ood:              ## leave-one-domain-out OOD slice (Data2txt held out)
	$(PY) scripts/40_robustness.py --run-id full

test:             ## fast offline test suite (incl. air-gap)
	$(PY) -m pytest -q

report:           ## assemble REPORT numbers + plots from runs/full
	$(PY) scripts/90_report.py --run-id full

clean:            ## remove run artifacts (keeps data cache + venv)
	rm -rf runs/*/scores.npz runs/*/*.png

all: data validate-crc slice test
