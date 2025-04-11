from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.utils
import json
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# Load student data from JSON file
def load_student_data():
    try:
        with open('student_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Save student data to JSON file
def save_student_data(students):
    with open('student_data.json', 'w') as f:
        json.dump(students, f, indent=4)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_student_data')
def get_student_data():
    students = load_student_data()
    return jsonify(students)

@app.route('/predict_student/<int:student_id>')
def predict_student(student_id):
    students = load_student_data()
    if student_id < len(students):
        student = students[student_id]
        # Generate suggestions based on student data
        suggestions = []
        
        if student['study_hours'] < 6:
            suggestions.append("Increase study hours to at least 6 hours per day")
        if student['attendance'] < 80:
            suggestions.append("Improve class attendance to above 80%")
        if student['participation_score'] < 7:
            suggestions.append("Participate more actively in class discussions")
            
        return jsonify({
            'prediction': student['performance'],
            'suggestions': suggestions
        })
    return jsonify({'error': 'Student not found'}), 404

@app.route('/delete_student/<int:student_id>', methods=['DELETE'])
def delete_student(student_id):
    students = load_student_data()
    if student_id < len(students):
        deleted_student = students.pop(student_id)
        save_student_data(students)
        return jsonify({'message': f'Student {deleted_student["name"]} deleted successfully'})
    return jsonify({'error': 'Student not found'}), 404

@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        students = load_student_data()
        student = {
            'name': request.form.get('name'),
            'study_hours': float(request.form.get('study_hours')),
            'attendance': float(request.form.get('attendance')),
            'previous_grades': float(request.form.get('previous_grades')),
            'participation_score': float(request.form.get('participation_score')),
            'socio_economic_status': request.form.get('socio_economic_status', 'Middle'),
            'performance': 0  # Will be calculated by the prediction model
        }
        students.append(student)
        save_student_data(students)
        flash('Student added successfully')
        return redirect(url_for('index'))
    return render_template('add_student.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Create a simple prediction model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Sample training data
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
def visualize():
    if not students:
        return jsonify({'plot': json.dumps({'data': [], 'layout': {}})})
    
    # Create visualization data
    names = [student['name'] for student in students]
    performances = [student.get('performance', 0) for student in students]
    
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
    app.run(debug=True) 