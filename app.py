from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly
import json
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    study_hours = db.Column(db.Float, nullable=False)
    attendance = db.Column(db.Float, nullable=False)
    previous_grades = db.Column(db.Float, nullable=False)
    participation_score = db.Column(db.Float, nullable=False)
    performance = db.Column(db.Float)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role')

        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))

        user = User(username=username, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Registration successful')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))

        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    students = Student.query.all()
    return render_template('dashboard.html', students=students)

@app.route('/add_student', methods=['GET', 'POST'])
@login_required
def add_student():
    if request.method == 'POST':
        student = Student(
            name=request.form.get('name'),
            study_hours=float(request.form.get('study_hours')),
            attendance=float(request.form.get('attendance')),
            previous_grades=float(request.form.get('previous_grades')),
            participation_score=float(request.form.get('participation_score'))
        )
        db.session.add(student)
        db.session.commit()
        flash('Student added successfully')
        return redirect(url_for('dashboard'))
    return render_template('add_student.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    data = request.get_json()
    
    # Create a simple prediction model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Sample training data (in a real application, you would use your actual training data)
    X_train = np.array([
        [5, 90, 80, 8],
        [3, 70, 65, 6],
        [7, 95, 85, 9],
        [4, 80, 75, 7],
        [6, 85, 80, 8]
    ])
    y_train = np.array([85, 65, 90, 75, 82])
    
    model.fit(X_train, y_train)
    
    # Make prediction
    X_test = np.array([[
        float(data['study_hours']),
        float(data['attendance']),
        float(data['previous_grades']),
        float(data['participation_score'])
    ]])
    
    prediction = model.predict(X_test)[0]
    
    # Generate suggestions based on prediction
    suggestions = []
    if prediction < 70:
        suggestions.append("Consider increasing study hours")
        suggestions.append("Focus on improving attendance")
        suggestions.append("Participate more in class activities")
    else:
        suggestions.append("Maintain current study habits")
        suggestions.append("Continue active participation")
        suggestions.append("Consider mentoring other students")
    
    return jsonify({
        'prediction': round(prediction, 2),
        'suggestions': suggestions
    })

@app.route('/visualize')
@login_required
def visualize():
    students = Student.query.all()
    
    # Create sample data for visualization
    names = [student.name for student in students]
    performances = [student.performance or 0 for student in students]
    
    data = [{
        'type': 'bar',
        'x': names,
        'y': performances,
        'marker': {
            'color': 'rgba(52, 152, 219, 0.7)'
        }
    }]
    
    layout = {
        'title': 'Student Performance Overview',
        'xaxis': {'title': 'Students'},
        'yaxis': {'title': 'Performance Score'},
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)'
    }
    
    return jsonify({'plot': json.dumps({'data': data, 'layout': layout})})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 