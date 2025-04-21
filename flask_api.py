from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from flask_cors import CORS
import json
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document
import re

app = Flask(__name__, 
    static_folder='static',
    template_folder='templates')
CORS(app)  # Enable CORS for all routes

# File path for permanent data storage
DATA_FILE = 'student_data.json'

# Initialize student data store
students = []

# Add these configurations after the app initialization
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load existing data if file exists
def load_data():
    global students
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as file:
                students = json.load(file)
                print(f"Loaded {len(students)} students from {DATA_FILE}")
                # Print first student to debug
                if students:
                    print(f"First student: {students[0]['name']}")
        except Exception as e:
            print(f"Error loading data: {e}")
            students = []
            # Add sample data if loading failed
            add_sample_data()
    else:
        print(f"Data file {DATA_FILE} not found. Starting with empty dataset.")
        students = []
        # Add sample data if no data exists
        add_sample_data()
        # Save sample data
        save_data()

# Save data to file
def save_data():
    try:
        with open(DATA_FILE, 'w') as file:
            json.dump(students, file, indent=4)
        print(f"Data saved to {DATA_FILE}")
    except Exception as e:
        print(f"Error saving data: {e}")

# Add sample data if no data exists
def add_sample_data():
    global students
    if not students:
        students.append({
            'name': 'John Smith',
            'study_hours': 6, 
            'attendance': 85,
            'previous_grades': 78,
            'participation_score': 7,
            'socio_economic_status': 'Middle',
            'extracurricular': 'Medium',
            'learning_style': 'Visual',
            'gender': 'Male',
            'parents_education': 'Graduate',
            'study_environment': 'Quiet',
            'parent_meeting_freq': 'Sometimes',
            'home_support': 'Moderate',
            'sleep_duration': 7.5,
            'stress_level': 'Moderate',
            'physical_activity': 'Moderate',
            'peer_group_quality': 'Good',
            'submission_timeliness': 'Good',
            'performance': 82
        })
        students.append({
            'name': 'Emma Johnson',
            'study_hours': 8, 
            'attendance': 92,
            'previous_grades': 88,
            'participation_score': 9,
            'socio_economic_status': 'High',
            'extracurricular': 'High',
            'learning_style': 'Reading/Writing',
            'gender': 'Female',
            'parents_education': 'Post-Graduate',
            'study_environment': 'Quiet',
            'parent_meeting_freq': 'Frequently',
            'home_support': 'High',
            'sleep_duration': 8,
            'stress_level': 'Low',
            'physical_activity': 'High',
            'peer_group_quality': 'Excellent',
            'submission_timeliness': 'Excellent',
            'performance': 90
        })
        students.append({
            'name': 'Michael Brown',
            'study_hours': 4, 
            'attendance': 70,
            'previous_grades': 65,
            'participation_score': 6,
            'socio_economic_status': 'Low',
            'extracurricular': 'Low',
            'learning_style': 'Kinesthetic',
            'gender': 'Male',
            'parents_education': 'Secondary',
            'study_environment': 'Noisy',
            'parent_meeting_freq': 'Rarely',
            'home_support': 'Low',
            'sleep_duration': 6,
            'stress_level': 'High',
            'physical_activity': 'Low',
            'peer_group_quality': 'Average',
            'submission_timeliness': 'Average',
            'performance': 68
        })

# Load data before defining routes
load_data()

@app.route('/')
def index():
    # Debug output
    print(f"Index route: {len(students)} students available")
    return render_template('index.html', students=students)

@app.route('/dashboard')
def dashboard():
    # Debug output
    print(f"Dashboard route: {len(students)} students available")
    return render_template('index.html', students=students)

@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        try:
            # Get form data
            student_data = request.form.to_dict()
            
            # Convert numeric fields
            numeric_fields = ['study_hours', 'attendance', 'previous_grades', 'participation_score']
            for field in numeric_fields:
                if field in student_data:
                    try:
                        student_data[field] = float(student_data[field])
                    except ValueError:
                        student_data[field] = 0
            
            # Make prediction
            prediction_result = make_prediction(student_data)
            student_data['performance'] = prediction_result['prediction']
            
            # Add to students list and save
            students.append(student_data)
            save_data()
            
            return jsonify({
                'success': True,
                'message': 'Student data added successfully'
            })
            
        except Exception as e:
            print(f"Error adding student: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            })
        
    # GET request - just show the form
    return render_template('add_student.html')

@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Convert numeric fields
        numeric_fields = ['study_hours', 'attendance', 'previous_grades', 'participation_score', 
                         'sleep_duration', 'stress_level', 'physical_activity']
        for field in numeric_fields:
            if field in data:
                try:
                    data[field] = float(data[field])
                except ValueError:
                    data[field] = 0
        
        # Make prediction
        result = make_prediction(data)
        
        # Get performance category
        category = get_performance_category(result['prediction'])
        
        # Prepare performance factors
        factors = [
            {
                'name': 'Study Hours',
                'value': f"{data['study_hours']} hrs",
                'impact': 'High' if data['study_hours'] >= 6 else 'Low'
            },
            {
                'name': 'Attendance',
                'value': f"{data['attendance']}%",
                'impact': 'High' if data['attendance'] >= 85 else 'Medium'
            },
            {
                'name': 'Previous Grades',
                'value': f"{data['previous_grades']}",
                'impact': 'High' if data['previous_grades'] >= 80 else 'Medium'
            },
            {
                'name': 'Participation',
                'value': f"{data['participation_score']}/10",
                'impact': 'High' if data['participation_score'] >= 7 else 'Medium'
            },
            {
                'name': 'Sleep Duration',
                'value': f"{data['sleep_duration']} hrs",
                'impact': 'Low' if data['sleep_duration'] < 7 else 'Medium'
            },
            {
                'name': 'Stress Level',
                'value': 'Low',
                'impact': 'Medium'
            },
            {
                'name': 'Physical Activity',
                'value': 'Moderate',
                'impact': 'Low'
            }
        ]
        
        # Feature importance data
        feature_importance = [
            {'feature': 'Previous Grades', 'importance': 0.20},
            {'feature': 'Study Hours', 'importance': 0.15},
            {'feature': 'Attendance', 'importance': 0.15},
            {'feature': 'Assignment Timeliness', 'importance': 0.10},
            {'feature': 'Peer Group Quality', 'importance': 0.07},
            {'feature': 'Study Environment', 'importance': 0.05},
            {'feature': 'Home Support', 'importance': 0.05},
            {'feature': 'Sleep Duration', 'importance': 0.05},
            {'feature': 'Participation', 'importance': 0.05},
            {'feature': 'Physical Activity', 'importance': 0.03},
            {'feature': 'Socio-economic Status', 'importance': 0.02},
            {'feature': 'Parents Education', 'importance': 0.02}
        ]
        
        # Study hours impact data (simulated relationship)
        study_hours_impact = [
            {'hours': 2, 'performance': 50},
            {'hours': 3, 'performance': 60},
            {'hours': 4, 'performance': 68},
            {'hours': 5, 'performance': 75},
            {'hours': 6, 'performance': 82},
            {'hours': 7, 'performance': 87},
            {'hours': 8, 'performance': 91},
            {'hours': 9, 'performance': 94},
            {'hours': 10, 'performance': 96}
        ]
        
        # Generate personalized suggestions
        suggestions = []
        if data['study_hours'] < 6:
            suggestions.append("Maintain your current study routine of 6 hours daily")
        if data['attendance'] >= 85:
            suggestions.append("Continue your excellent attendance record to stay on top of course material")
        if data['sleep_duration'] < 7:
            suggestions.append("Consider increasing sleep duration to 7-8 hours for optimal cognitive performance")
        suggestions.extend([
            "Develop a structured study plan with specific goals for each session",
            "Use the Pomodoro technique (25-minute focused sessions followed by 5-minute breaks)",
            "Practice retrieval-based studying rather than passive re-reading",
            "Create concept maps to visualize connections between different topics",
            "Teach concepts to others to solidify your understanding"
        ])
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'category': category,
            'factors': factors,
            'suggestions': suggestions,
            'featureImportance': feature_importance,
            'studyHoursImpact': study_hours_impact
        })
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/about')
def about():
    # Render the index.html template with the about section visible
    return render_template('about.html') if os.path.exists('templates/about.html') else render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/api/students', methods=['GET'])
def get_students():
    return jsonify(students)

@app.route('/api/students/<int:student_index>', methods=['DELETE'])
def delete_student(student_index):
    try:
        if 0 <= student_index < len(students):
            deleted_student = students.pop(student_index)
            save_data()  # Save changes to file
            return jsonify({
                "message": "Student deleted successfully",
                "deleted_student": deleted_student
            }), 200
        else:
            return jsonify({"error": "Student not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/students', methods=['POST'])
def add_student_api():
    try:
        student_data = request.json
        
        # Convert numeric fields
        numeric_fields = ['study_hours', 'attendance', 'previous_grades', 'participation_score', 'sleep_duration', 'stress_level', 'physical_activity']
        for field in numeric_fields:
            if field in student_data:
                try:
                    student_data[field] = float(student_data[field])
                except (ValueError, TypeError):
                    student_data[field] = 0
        
        # If performance is not provided, calculate it
        if 'performance' not in student_data:
            prediction_result = make_prediction(student_data)
            student_data['performance'] = prediction_result['prediction']
        
        # Add student to the list
        students.append(student_data)
        
        # Save changes to file
        save_data()
        
        print(f"Added new student: {student_data['name']}")  # Debug log
        
        return jsonify({
            "message": "Student added successfully",
            "student": student_data
        }), 201
        
    except Exception as e:
        print(f"Error adding student: {str(e)}")  # Debug log
        return jsonify({
            "error": f"Failed to add student: {str(e)}"
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify(calculate_stats())

@app.route('/visualize')
def visualize():
    """Route for visualization data that's called from the dashboard"""
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

def make_prediction(student_data):
    # Convert categorical variables to numeric
    socio_economic_map = {'low': 0, 'middle': 1, 'high': 2}
    activity_level_map = {'none': 0, 'low': 1, 'moderate': 2, 'high': 3}
    learning_style_map = {'visual': 0, 'auditory': 1, 'kinesthetic': 2, 'reading_writing': 3}
    gender_map = {'male': 0, 'female': 1, 'other': 2}
    education_level_map = {'primary': 0, 'secondary': 1, 'graduate': 2, 'post_graduate': 3}
    environment_map = {'noisy': 0, 'moderate': 1, 'quiet': 2}
    frequency_map = {'never': 0, 'rarely': 1, 'sometimes': 2, 'often': 3}
    quality_map = {'low': 0, 'moderate': 1, 'high': 2}
    # Map for assignment timeliness
    timeliness_map = {'never': 0, 'often_late': 25, 'sometimes_late': 50, 'always_on_time': 100}

    # Extract features and convert to appropriate numeric values
    features = {
        'study_hours': float(student_data.get('study_hours', 0)),
        'attendance': float(student_data.get('attendance', 0)),
        'previous_grades': float(student_data.get('previous_grades', 0)),
        'participation_score': float(student_data.get('participation_score', 0)),
        'assignment_timeliness': timeliness_map.get(student_data.get('assignment_timeliness', '').lower(), 0),
        'socio_economic_status': socio_economic_map.get(student_data.get('socio_economic_status', '').lower(), 1),
        'extracurricular': activity_level_map.get(student_data.get('extracurricular', '').lower(), 1),
        'learning_style': learning_style_map.get(student_data.get('learning_style', '').lower(), 0),
        'gender': gender_map.get(student_data.get('gender', '').lower(), 0),
        'parents_education': education_level_map.get(student_data.get('parents_education', '').lower(), 1),
        'study_environment': environment_map.get(student_data.get('study_environment', '').lower(), 1),
        'parent_meeting_freq': frequency_map.get(student_data.get('parent_meeting_freq', '').lower(), 1),
        'home_support': quality_map.get(student_data.get('home_support', '').lower(), 1),
        'sleep_duration': float(student_data.get('sleep_duration', 7)),
        'stress_level': float(student_data.get('stress_level', 5)),
        'physical_activity': float(student_data.get('physical_activity', 0)),
        'peer_academic_quality': quality_map.get(student_data.get('peer_academic_quality', '').lower(), 1)
    }

    # Simple prediction model (you can replace this with a more sophisticated model)
    weights = {
        'study_hours': 0.15,
        'attendance': 0.15,
        'previous_grades': 0.20,
        'participation_score': 0.05,
        'assignment_timeliness': 0.10,
        'socio_economic_status': 0.02,
        'extracurricular': 0.02,
        'learning_style': 0.02,
        'parents_education': 0.02,
        'study_environment': 0.05,
        'parent_meeting_freq': 0.02,
        'home_support': 0.05,
        'sleep_duration': 0.05,
        'stress_level': -0.05,
        'physical_activity': 0.03,
        'peer_academic_quality': 0.07
    }

    # Normalize features
    normalized_features = {}
    for key, value in features.items():
        if key in ['study_hours']:
            normalized_features[key] = min(value / 12.0, 1.0)  # Max 12 hours per day
        elif key in ['attendance', 'assignment_timeliness', 'previous_grades']:
            normalized_features[key] = value / 100.0
        elif key in ['participation_score', 'stress_level']:
            normalized_features[key] = value / 10.0
        elif key in ['sleep_duration']:
            normalized_features[key] = min(value / 12.0, 1.0)  # Max 12 hours
        elif key in ['physical_activity']:
            normalized_features[key] = min(value / 20.0, 1.0)  # Max 20 hours
        else:
            # Already normalized categorical variables
            normalized_features[key] = value / 3.0  # Most categorical variables are 0-3

    # Calculate weighted sum
    prediction = sum(weights[key] * normalized_features[key] for key in weights.keys()) * 100

    # Ensure prediction is between 0 and 100
    prediction = max(0, min(100, prediction))

    # Generate suggestions based on the features
    suggestions = []
    
    if features['study_hours'] < 5:
        suggestions.append("Consider increasing daily study hours to improve performance")
    if features['attendance'] < 80:
        suggestions.append("Try to improve attendance to better understand the subjects")
    if features['sleep_duration'] < 7:
        suggestions.append("Getting more sleep could help improve focus and learning")
    if features['stress_level'] > 7:
        suggestions.append("Consider stress management techniques to improve learning efficiency")
    if features['physical_activity'] < 3:
        suggestions.append("Including more physical activity could help reduce stress and improve focus")
    if features['assignment_timeliness'] < 75:
        suggestions.append("Try to submit assignments on time to improve overall performance")
    if features['participation_score'] < 7:
        suggestions.append("More active participation in class could help better understand the subjects")

    return {
        'prediction': round(prediction, 2),
        'suggestions': suggestions
    }

def get_performance_category(score):
    """Convert numerical score to category"""
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 60:
        return "Average"
    else:
        return "Needs Improvement"

def calculate_stats():
    """Calculate student statistics"""
    if not students:
        return {
            "total_students": 0,
            "average_performance": 0,
            "performance_distribution": {
                "Excellent": 0,
                "Good": 0, 
                "Average": 0,
                "Needs Improvement": 0
            }
        }
    
    # Calculate average performance
    total_students = len(students)
    performances = [s.get('performance', 0) for s in students]
    average_performance = sum(performances) / total_students if total_students > 0 else 0
    average_performance = round(average_performance, 1)
    
    # Calculate performance distribution
    performance_distribution = {
        "Excellent": 0,
        "Good": 0,
        "Average": 0,
        "Needs Improvement": 0
    }
    
    for p in performances:
        category = get_performance_category(p)
        performance_distribution[category] += 1
    
    return {
        "total_students": total_students,
        "average_performance": average_performance,
        "performance_distribution": performance_distribution
    }

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def analyze_resume(text):
    # Initialize scores
    content_score = 0
    format_score = 0
    style_score = 0
    skills_score = 0
    recommendations = []

    # Content Analysis
    sections = ['education', 'experience', 'skills', 'projects']
    found_sections = 0
    for section in sections:
        if re.search(section, text.lower()):
            found_sections += 1
    content_score = (found_sections / len(sections)) * 100

    # Format Analysis
    lines = text.split('\n')
    format_score = min(100, (len(lines) / 40) * 100)  # Assume ideal length is 40 lines

    # Style Analysis
    action_verbs = ['developed', 'implemented', 'created', 'managed', 'led', 'designed', 'analyzed']
    verb_count = sum(1 for verb in action_verbs if verb in text.lower())
    style_score = min(100, (verb_count / 5) * 100)  # Assume 5 action verbs is ideal

    # Skills Analysis
    technical_skills = ['python', 'java', 'javascript', 'sql', 'react', 'node', 'machine learning']
    found_skills = sum(1 for skill in technical_skills if skill in text.lower())
    skills_score = min(100, (found_skills / len(technical_skills)) * 100)

    # Generate Recommendations
    if content_score < 70:
        recommendations.append("Add more details to key sections (Education, Experience, Skills, Projects)")
    if format_score < 70:
        recommendations.append("Optimize resume length and structure")
    if style_score < 70:
        recommendations.append("Use more action verbs to describe your achievements")
    if skills_score < 70:
        recommendations.append("Include more relevant technical skills")

    # Calculate overall score
    overall_score = int((content_score + format_score + style_score + skills_score) / 4)

    return {
        'score': overall_score,
        'contentScore': int(content_score),
        'formatScore': int(format_score),
        'styleScore': int(style_score),
        'skillsScore': int(skills_score),
        'recommendations': recommendations
    }

@app.route('/resume_analysis')
def resume_analysis():
    return render_template('resume_analysis.html')

@app.route('/analyze_resume', methods=['POST'])
def analyze_resume_api():
    if 'resume' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'})
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text based on file type
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(filepath)
        else:
            text = extract_text_from_docx(filepath)
        
        # Analyze the resume
        analysis_result = analyze_resume(text)
        
        # Clean up the temporary file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'ats_score': analysis_result['score'],
            'ats_verdict': get_performance_category(analysis_result['score']),
            'ats_summary': analysis_result['recommendations'][0] if analysis_result['recommendations'] else 'No summary available',
            'technical_skills': analysis_result['skillsScore'] > 70,
            'soft_skills': analysis_result['skillsScore'] > 70,
            'education': analysis_result['contentScore'] > 70,
            'experience': analysis_result['formatScore'] > 70,
            'skills_match_percentage': analysis_result['skillsScore'] > 70,
            'recommendations': analysis_result['recommendations']
        })
        
    except Exception as e:
        print(f"Error analyzing resume: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/student_data')
def student_data():
    """Route to display student data page"""
    return render_template('student_data.html', students=students)

if __name__ == '__main__':
    # Create static folder if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Create static/css folder if it doesn't exist
    if not os.path.exists('static/css'):
        os.makedirs('static/css')
    
    # Create static/js folder if it doesn't exist 
    if not os.path.exists('static/js'):
        os.makedirs('static/js')
    
    app.run(debug=True, port=5000) 