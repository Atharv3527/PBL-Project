# Change to the project directory
Set-Location "C:\PBL Project"

# Activate virtual environment
.\venv\Scripts\activate

# Install required packages
pip install flask flask-sqlalchemy pandas numpy scikit-learn plotly flask-cors PyPDF2 python-docx

# Check if flask_api.py exists
if (-not (Test-Path "flask_api.py")) {
    Write-Host "Error: flask_api.py not found. Please make sure it exists in the project directory."
    exit 1
}

# Start Flask API in the background
Write-Host "Starting Flask API..."
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "flask_api.py"

# Wait for Flask API to start
Write-Host "Waiting for Flask API to initialize..."
Start-Sleep -Seconds 3

# Navigate to client directory
Write-Host "Setting up React application..."
Set-Location ".\client"

# Create a temporary batch file to run React
@"
cd C:\PBL Project\client
npm start
"@ | Out-File -FilePath "run_react.bat" -Encoding ascii

# Run the batch file in a new window
Write-Host "Starting React client..."
Start-Process -FilePath "run_react.bat" -WindowStyle Normal

# Wait for React to start
Write-Host "Waiting for React to start..."
Start-Sleep -Seconds 5

# Open browser
Write-Host "Opening browser..."
Start-Process "http://localhost:3000"

# Navigate back to project root
Set-Location "C:\PBL Project"

Write-Host "Done! Your application should be running."
Write-Host "- Flask API: http://127.0.0.1:5000"
Write-Host "- React App: http://localhost:3000" 