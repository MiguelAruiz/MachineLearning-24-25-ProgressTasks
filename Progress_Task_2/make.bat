@echo off

rem Inicia el servidor MLflow en el puerto 8080
start cmd /k "mlflow server --port 8080"

rem Esperar unos segundos para asegurarse de que el servidor esté en ejecución (ajusta según sea necesario)
timeout /t 5 >nul

rem Ejecuta el script de Python
python3 mlflow/test.py
