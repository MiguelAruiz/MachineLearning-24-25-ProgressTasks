from types import ModuleType
from typing import Callable, Tuple

import mlflow
import test1
import test2

from create_dataset import Dataset
from run_test import run_test

tests: list[ModuleType] = [ test2]


def main(argv: list[str] = []):
    # FIXME duplicated code with Progress_Task_2/mlflow/test.py:89
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    experiment_name = "Whole dataset + correlation"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    # Create a new MLflow Experiment
    mlflow.set_experiment(experiment_name)
    # FIXME PROBABLY BEST TO GET THIS IN A FUNCTION
    d = Dataset()
    print("Running Tests...")
    for t in tests:
        print(f"Running test {t}:")
        run_test(t, d)
    print("Run Finished.")


if __name__ == "__main__":
    main()
