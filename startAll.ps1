# Change to the project directory
Set-Location "C:\PBL Project"

# Activate virtual environment
.\venv\Scripts\activate

# Install required packages
pip install flask flask-sqlalchemy pandas numpy scikit-learn plotly flask-cors PyPDF2 python-docx

# Build the React app
Write-Host "Building React app..."
Set-Location .\client
npm run build
Set-Location ..

# Copy React build files to Flask static directory
Write-Host "Copying React build files to Flask static directory..."
if (Test-Path .\static) { Remove-Item .\static -Recurse -Force }
Copy-Item .\client\build\* .\static -Recurse

# Check if flask_api.py exists
if (-not (Test-Path "flask_api.py")) {
    Write-Host "Error: flask_api.py not found. Please make sure it exists in the project directory."
    exit 1
}

# Start Flask API in the background with the --no-browser flag
Write-Host "Starting Flask API..."
$env:FLASK_NO_OPEN_BROWSER = "true"
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "flask_api.py"

Write-Host "Done! Your application should be running."
Write-Host "- Flask API running on: http://127.0.0.1:5000"
Write-Host "Access your updated website at: http://127.0.0.1:5000" 