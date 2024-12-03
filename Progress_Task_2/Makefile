MLFLOW = mlflow server
MLFLOW_OPTIONS = --port
MLFLOW_PORT = 8080
DETATCH = &

PYTHON = python3
TESTS_ROUTE = mlflow/test.py

CLEAN = rm -rf
OUTPUT_FILE = mlflow/predictions.csv

INSTALL_COMMAND = pip install
INSTALL_FROM_REQUIREMENTS = -r requirements.txt

all: clean install mlflow tests

mlflow:
	$(MLFLOW) $(MLFLOW_OPTIONS) $(MLFLOW_PORT)

tests:
	$(PYTHON) $(TESTS_ROUTE)

clean:
	$(CLEAN) $(OUTPUT_FILE)

install:
	$(INSTALL_COMMAND) $(INSTALL_FROM_REQUIREMENTS)

.PHONY: all clean