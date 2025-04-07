# Student Performance Prediction System

A web application that predicts student performance based on various factors and provides personalized suggestions for improvement.

## Features

- Add student data
- Predict student performance
- View performance visualizations
- Get personalized improvement suggestions

## Stack

- **Frontend**: React, React Bootstrap, Chart.js
- **Backend**: Flask REST API
- **Data Science**: scikit-learn, pandas, numpy
- **Visualization**: react-chartjs-2, plotly

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Atharv3527/PerformancePredictor.git
cd PerformancePredictor
```

2. Create and activate virtual environment for Python:
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac
```

3. Install backend dependencies:
```bash
pip install -r requirements.txt
```

4. Install frontend dependencies:
```bash
cd client
npm install
cd ..
```

## Usage

### Method 1: Run Backend and Frontend Separately

1. Start the Flask backend:
```bash
python flask_api.py
```

2. In a separate terminal, start the React frontend:
```bash
cd client
npm start
```

3. Open your web browser and go to:
```
http://localhost:3000
```

### Method 2: Run Using npm Scripts (requires npm installed globally)

1. Install concurrently package:
```bash
npm install
```

2. Start both backend and frontend:
```bash
npm start
```

## Requirements

- Python 3.8+
- Node.js 14+
- npm 6+

## License

MIT License 