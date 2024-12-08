MLFLOW = mlflow server
MLFLOW_OPTIONS = --port
MLFLOW_PORT = 8080
DETATCH = &

PYTHON = python3
TESTS_ROUTE = mlflow/test.py

MOVE_TO = cd
AND_EXECUTE = &&
PREVIOUS_DIR = ..
PROGRESS_TASK2_DIR = Progress_Task_2

CLEAN = rm -rf
OUTPUT_FILE = mlflow/predictions.csv

INSTALL_COMMAND = pip install
INSTALL_FROM_REQUIREMENTS = -r requirements.txt

all: install tests
setup: install clean mlflow

mlflow:
	$(MOVE_TO) $(PREVIOUS_DIR) $(AND_EXECUTE) $(MLFLOW) $(MLFLOW_OPTIONS) $(MLFLOW_PORT)

tests:
	pwd && $(PYTHON) $(TESTS_ROUTE)

clean:
	$(CLEAN) $(OUTPUT_FILE)

install:
	$(INSTALL_COMMAND) $(INSTALL_FROM_REQUIREMENTS)

.PHONY: all clean mlflow