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
        # Handle form submission
        student_data = request.form.to_dict()
        
        # Convert numeric values to appropriate types
        numeric_fields = [
            'study_hours', 
            'attendance', 
            'previous_grades', 
            'participation_score',
            'sleep_duration',
            'stress_level',
            'physical_activity'
        ]
        
        for field in numeric_fields:
            if field in student_data and student_data[field]:
                try:
                    student_data[field] = float(student_data[field])
                except ValueError:
                    student_data[field] = 0  # Default value if conversion fails
        
        # Calculate performance using prediction model
        prediction_result = make_prediction(student_data)
        student_data['performance'] = prediction_result['prediction']
        
        # Add to students list
        students.append(student_data)
        save_data()  # Save to file
        
        # Redirect to dashboard
        return redirect(url_for('dashboard'))
        
    # GET request - just show the form
    return render_template('add_student.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    prediction_result = None
    student_data = {}
    
    if request.method == 'POST':
        # Handle form submission
        student_data = request.form.to_dict()
        
        # Convert numeric values to float
        numeric_fields = [
            'study_hours', 
            'attendance', 
            'previous_grades', 
            'participation_score',
            'sleep_duration',
            'stress_level',
            'physical_activity'
        ]
        
        for field in numeric_fields:
            if field in student_data and student_data[field]:
                try:
                    student_data[field] = float(student_data[field])
                except (ValueError, TypeError):
                    student_data[field] = 0  # Default value if conversion fails
        
        # Get prediction
        prediction_result = make_prediction(student_data)
        
        # Convert any non-string values back to strings for the template
        for key, value in student_data.items():
            if not isinstance(value, str):
                student_data[key] = str(value)
    
    # GET request or after POST processing
    return render_template('predict.html', prediction=prediction_result, student_data=student_data)

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
    student_data = request.json
    
    # If performance is not provided, calculate it
    if 'performance' not in student_data:
        prediction_result = make_prediction(student_data)
        student_data['performance'] = prediction_result['prediction']
    
    students.append(student_data)
    save_data()  # Save changes to file
    return jsonify({"message": "Student added successfully", "student": student_data}), 201

@app.route('/api/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # For GET requests, return empty response with example structure
        return jsonify({
            "prediction": 0,
            "performance_category": "Example",
            "suggestions": ["This is an example API. Please use POST method with student data."],
            "feature_importance": {
                "study_hours": 0.3,
                "attendance": 0.25,
                "previous_grades": 0.2
            }
        })
    
    # For POST requests
    if request.is_json:
        # JSON data from fetch API
        student_data = request.json
    else:
        # Form data
        student_data = request.form.to_dict()
        
        # Convert numeric values to float
        for key in ['study_hours', 'attendance', 'previous_grades', 'participation_score']:
            if key in student_data and student_data[key]:
                try:
                    student_data[key] = float(student_data[key])
                except ValueError:
                    pass  # Keep as string if conversion fails
    
    # Get prediction
    try:
        prediction_result = make_prediction(student_data)
        return jsonify(prediction_result)
    except Exception as e:
        print(f"Error making prediction: {e}")
        return jsonify({
            "error": "Failed to make prediction",
            "message": str(e)
        }), 400

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
        return jsonify({
            'success': False,
            'error': 'No file provided'
        })
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        })
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Extract text based on file type
            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            else:  # docx
                text = extract_text_from_docx(file_path)
            
            # Analyze the resume text
            analysis_result = {
                'success': True,
                'skills': {
                    'technical': extract_technical_skills(text),
                    'soft': extract_soft_skills(text)
                },
                'education': extract_education(text),
                'experience': extract_experience(text),
                'recommendations': generate_recommendations(text)
            }
            
            # Clean up the uploaded file
            os.remove(file_path)
            
            return jsonify(analysis_result)
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    return jsonify({
        'success': False,
        'error': 'Invalid file type'
    })

def extract_technical_skills(text):
    technical_skills = [
        'Python', 'Java', 'JavaScript', 'C++', 'SQL',
        'Machine Learning', 'Data Analysis', 'Web Development',
        'React', 'Node.js', 'Docker', 'AWS', 'Git'
    ]
    found_skills = []
    for skill in technical_skills:
        if skill.lower() in text.lower():
            found_skills.append(skill)
    return found_skills

def extract_soft_skills(text):
    soft_skills = [
        'Communication', 'Leadership', 'Problem Solving',
        'Team Work', 'Time Management', 'Project Management',
        'Critical Thinking', 'Adaptability', 'Creativity'
    ]
    found_skills = []
    for skill in soft_skills:
        if skill.lower() in text.lower():
            found_skills.append(skill)
    return found_skills

def extract_education(text):
    # Simple education extraction (can be enhanced with regex)
    education = []
    degrees = ['Bachelor', 'Master', 'PhD', 'BSc', 'MSc', 'MBA']
    lines = text.split('\n')
    
    for line in lines:
        for degree in degrees:
            if degree.lower() in line.lower():
                education.append({
                    'degree': line.strip(),
                    'institution': 'Institution',  # This could be enhanced with better parsing
                    'year': 'Year'  # This could be enhanced with better parsing
                })
                break
    
    return education

def extract_experience(text):
    # Simple experience extraction (can be enhanced with regex)
    experience = []
    keywords = ['experience', 'work', 'position', 'job']
    lines = text.split('\n')
    
    current_exp = None
    for line in lines:
        if any(keyword in line.lower() for keyword in keywords):
            if current_exp:
                experience.append(current_exp)
            current_exp = {
                'position': line.strip(),
                'company': 'Company',  # This could be enhanced with better parsing
                'duration': 'Duration',  # This could be enhanced with better parsing
                'description': 'Description'  # This could be enhanced with better parsing
            }
    
    if current_exp:
        experience.append(current_exp)
    
    return experience

def generate_recommendations(text):
    recommendations = []
    
    # Check for technical skills
    if len(extract_technical_skills(text)) < 5:
        recommendations.append("Consider adding more technical skills to your resume")
    
    # Check for soft skills
    if len(extract_soft_skills(text)) < 3:
        recommendations.append("Include more soft skills to show your interpersonal abilities")
    
    # Check for education
    if len(extract_education(text)) == 0:
        recommendations.append("Add your educational background with degrees and institutions")
    
    # Check for experience
    if len(extract_experience(text)) == 0:
        recommendations.append("Include your work experience with detailed responsibilities")
    
    # General recommendations
    recommendations.extend([
        "Use action verbs to describe your achievements",
        "Quantify your achievements with numbers and metrics",
        "Ensure your resume is properly formatted and easy to read"
    ])
    
    return recommendations

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