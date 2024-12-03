MLFLOW = mlflow server
MLFLOW_OPTIONS = --port
MLFLOW_PORT = 8080
DETATCH = &

PYTHON = python3
TESTS_ROUTE = mlflow/test.py


CLEAN = rm -rf
OUTPUT_FILE = mlflow/predictions.csv

all: clean mlflow tests

mlflow:
	$(MLFLOW) $(MLFLOW_OPTIONS) $(MLFLOW_PORT)

tests:
	$(PYTHON) $(TESTS_ROUTE)

clean:
	$(CLEAN) $(OUTPUT_FILE)

.PHONY: all clean