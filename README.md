# Student Performance Prediction System

A web application that predicts student performance based on various academic and behavioral factors.

## Features

- User Authentication (Login/Register)
- Student Data Management
- Performance Prediction using Machine Learning
- Data Visualization
- Interactive Dashboard

## Technologies Used

- Backend: Flask (Python)
- Frontend: HTML, CSS, JavaScript, Bootstrap 5
- Database: SQLite
- Machine Learning: scikit-learn
- Data Visualization: Plotly, Matplotlib, Seaborn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Atharv3527/PerformancePredictor.git
cd PerformancePredictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
PerformancePredictor/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
│   ├── base.html      # Base template
│   ├── login.html     # Login page
│   ├── register.html  # Registration page
│   ├── dashboard.html # Dashboard
│   └── add_student.html # Add student form
└── instance/          # Database and instance files
```

## Usage

1. Register a new account or login with existing credentials
2. Add student data through the dashboard
3. View performance predictions and visualizations
4. Analyze student performance trends

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 