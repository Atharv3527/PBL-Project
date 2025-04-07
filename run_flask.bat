@echo off
echo Starting Student Performance Prediction System...

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Please create it first.
    exit /b 1
)

:: Activate virtual environment and run Flask app
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install required packages
echo Checking packages...
pip install flask flask-sqlalchemy pandas numpy scikit-learn plotly flask-cors werkzeug

:: Run Flask app
echo Starting Flask app...
python flask_api.py

:: Deactivate virtual environment when Flask exits
call deactivate 