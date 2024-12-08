@echo off

rem Remove the file mlflow/predictions.csv if it exists
if exist mlflow/predictions.csv del mlflow/predictions.csv

rem Install Dependencies
pip install -r requirements.txt

rem Inicia el servidor MLflow en el puerto 8080
start cmd "mlflow server --port 8080"

rem Esperar unos segundos para asegurarse de que el servidor esté en ejecución (ajusta según sea necesario)
timeout /t 5 >nul

rem Ejecuta el script de Python
python3 mlflow/test.py
