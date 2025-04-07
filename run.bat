@echo off
echo Starting Student Performance Prediction System...

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Please create it first.
    exit /b 1
)

:: Start Flask API in a new window
echo Starting Flask API...
start cmd /k "cd /d C:\PBL Project && venv\Scripts\activate.bat && python flask_api.py"

:: Wait for Flask API to start
echo Waiting for Flask API to initialize...
timeout /t 3 /nobreak > nul

:: Start React app in a new window
echo Starting React client...
start cmd /k "cd /d C:\PBL Project\client && npm start"

:: Wait for React to start
echo Waiting for React to start...
timeout /t 5 /nobreak > nul

:: Open browser
echo Opening browser...
start http://localhost:3000

echo.
echo Done! Your application should be running.
echo - Flask API: http://127.0.0.1:5000
echo - React App: http://localhost:3000
echo.
echo Press any key to exit this window...
pause > nul 