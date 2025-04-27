# Change to the project directory
Set-Location "C:\PBL-Project"

# Install required packages
pip install -r requirements.txt

# Start Flask server
python flask_api.py

Write-Host "Done! Your application should be running."
Write-Host "- Flask API running on: http://127.0.0.1:5000"
Write-Host "Access your updated website at: http://127.0.0.1:5000" 