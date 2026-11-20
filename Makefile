CONFIG ?= configs/default.yaml
SOILCLF = soilclf --config $(CONFIG)

.PHONY: install install-train audit baseline train evaluate export all test app clean

install:          ## core + dev deps (no TensorFlow)
	pip install -e ".[dev]"

install-train:    ## everything, including TensorFlow and Streamlit
	pip install -e ".[train,app,dev]"

audit:            ## de-duplicate, flag label conflicts, write leakage-free split
	$(SOILCLF) audit

baseline: audit   ## handcrafted color/texture features + scikit-learn
	$(SOILCLF) baseline

train: audit      ## MobileNetV2 transfer learning (head, then fine-tune)
	$(SOILCLF) train

evaluate:         ## val/test metrics with bootstrap 95% CIs
	$(SOILCLF) evaluate

export:           ## TFLite fp32/fp16/int8 + size, accuracy, latency benchmark
	$(SOILCLF) export

all: baseline train evaluate export

test:
	pytest -q

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf artifacts .pytest_cache
