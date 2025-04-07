# Change to the project directory
Set-Location "C:\PBL Project"

# Activate virtual environment
.\venv\Scripts\activate

# Install required packages
pip install flask flask-sqlalchemy pandas numpy scikit-learn plotly

# Start the Flask application
python app.py 