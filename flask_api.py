from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, flash
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
import logging
from datetime import datetime
import plotly.express as px
import plotly.utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
    static_folder='static',
    template_folder='templates')
CORS(app)  # Enable CORS for all routes
app.config['SECRET_KEY'] = 'your-secret-key'

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

# Create static folders if they don't exist
if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists('static/css'):
    os.makedirs('static/css')
if not os.path.exists('static/js'):
    os.makedirs('static/js')

# Input validation functions
def validate_student_data(data):
    required_fields = ['name', 'study_hours', 'attendance', 'previous_grades', 
                      'participation_score', 'socio_economic_status']
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate numeric fields
    try:
        float(data['study_hours'])
        float(data['attendance'])
        float(data['previous_grades'])
        float(data['participation_score'])
    except (ValueError, TypeError):
        raise ValueError("Invalid numeric values provided")
    
    # Validate ranges
    if not (0 <= float(data['attendance']) <= 100):
        raise ValueError("Attendance must be between 0 and 100")
    if not (0 <= float(data['previous_grades']) <= 100):
        raise ValueError("Previous grades must be between 0 and 100")
    if not (0 <= float(data['participation_score']) <= 10):
        raise ValueError("Participation score must be between 0 and 10")
    
    return True

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
        print(f"Saved {len(students)} students to {DATA_FILE}")
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
    try:
        load_data()
        return render_template('index.html', students=students)
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        flash("An error occurred while loading the page.", "error")
        return render_template('index.html', students=[])

@app.route('/dashboard')
def dashboard():
    try:
        load_data()
        return render_template('index.html', students=students)
    except Exception as e:
        logger.error(f"Error in dashboard route: {e}")
        flash("An error occurred while loading the dashboard.", "error")
        return render_template('index.html', students=[])

@app.route('/resume_analysis')
def resume_analysis():
    return render_template('resume_analysis.html')

@app.route('/about')
def about():
    return render_template('about.html') if os.path.exists('templates/about.html') else render_template('index.html')

@app.route('/student_data')
def student_data():
    try:
        load_data()
        return render_template('student_data.html', students=students)
    except Exception as e:
        logger.error(f"Error in student_data route: {e}")
        flash("An error occurred while loading student data.", "error")
        return render_template('student_data.html', students=[])

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/api/students', methods=['GET'])
def get_students():
    try:
        load_data()
        return jsonify(students)
    except Exception as e:
        logger.error(f"Error getting students: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while fetching students'
        }), 500

@app.route('/api/students', methods=['POST'])
def add_student():
    try:
        data = request.get_json()
        validate_student_data(data)
        
        # Load existing data
        load_data()
        
        # Add new student
        student = {
            'id': len(students) + 1,
            'name': data['name'],
            'study_hours': float(data['study_hours']),
            'attendance': float(data['attendance']),
            'previous_grades': float(data['previous_grades']),
            'participation_score': float(data['participation_score']),
            'socio_economic_status': data['socio_economic_status'],
            'prediction': predict_performance(data)
        }
        
        students.append(student)
        save_data()
        
        return jsonify({
            'success': True,
            'message': 'Student added successfully',
            'student': student
        })
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error adding student: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while adding the student'
        }), 500

@app.route('/api/students/<int:student_id>', methods=['DELETE'])
def delete_student(student_id):
    try:
        load_data()
        student = next((s for s in students if s['id'] == student_id), None)
        if student:
            students.remove(student)
            save_data()
            return jsonify({
                'success': True,
                'message': 'Student deleted successfully'
            })
        return jsonify({
            'success': False,
            'error': 'Student not found'
        }), 404
    except Exception as e:
        logger.error(f"Error deleting student: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while deleting the student'
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_performance():
    try:
        data = request.get_json()
        validate_student_data(data)
        
        prediction = predict_performance(data)
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while making the prediction'
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

def analyze_resume_content(text):
    # Initialize scores
    content_score = 0
    format_score = 0
    style_score = 0
    skills_score = 0
    recommendations = []

    # Content Analysis
    sections = ['education', 'experience', 'skills', 'projects', 'summary', 'achievements']
    found_sections = 0
    for section in sections:
        if re.search(section, text.lower()):
            found_sections += 1
    content_score = (found_sections / len(sections)) * 100

    # Format Analysis
    lines = text.split('\n')
    format_score = min(100, (len(lines) / 40) * 100)  # Assume ideal length is 40 lines

    # Style Analysis
    action_verbs = ['developed', 'implemented', 'created', 'managed', 'led', 'designed', 'analyzed', 
                   'optimized', 'improved', 'increased', 'decreased', 'reduced', 'enhanced', 'streamlined']
    verb_count = sum(1 for verb in action_verbs if verb in text.lower())
    style_score = min(100, (verb_count / 5) * 100)  # Assume 5 action verbs is ideal

    # Skills Analysis
    technical_skills = ['python', 'java', 'javascript', 'sql', 'react', 'node', 'machine learning',
                       'data analysis', 'cloud computing', 'aws', 'azure', 'docker', 'kubernetes',
                       'agile', 'scrum', 'git', 'ci/cd', 'rest api', 'microservices']
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

    # Additional specific recommendations
    if not re.search(r'\b(summary|profile|objective)\b', text.lower()):
        recommendations.append("Add a professional summary or objective statement")
    if not re.search(r'\b(achievements|accomplishments)\b', text.lower()):
        recommendations.append("Include a section for key achievements and accomplishments")
    if not re.search(r'\b(projects|portfolio)\b', text.lower()):
        recommendations.append("Add a projects section to showcase your work")
    if not re.search(r'\b(certifications|certificates)\b', text.lower()):
        recommendations.append("Include relevant certifications and training")
    if not re.search(r'\b(volunteer|community)\b', text.lower()):
        recommendations.append("Consider adding volunteer or community involvement")
    if not re.search(r'\b(languages|bilingual)\b', text.lower()):
        recommendations.append("List any additional languages you speak")
    if not re.search(r'\b(publications|papers)\b', text.lower()):
        recommendations.append("Include any publications or research papers")

    # Additional detailed recommendations
    if not re.search(r'\b(quantified|metrics|numbers|percentages)\b', text.lower()):
        recommendations.append("Add quantifiable achievements with specific numbers and metrics")
    if not re.search(r'\b(keywords|industry|specific|terms)\b', text.lower()):
        recommendations.append("Include industry-specific keywords from the job description")
    if not re.search(r'\b(soft skills|interpersonal|communication|leadership)\b', text.lower()):
        recommendations.append("Highlight your soft skills and interpersonal abilities")
    if not re.search(r'\b(education|degree|university|college)\b', text.lower()):
        recommendations.append("Ensure your education section is complete and up-to-date")
    if not re.search(r'\b(experience|work history|employment)\b', text.lower()):
        recommendations.append("Provide detailed work experience with responsibilities and achievements")
    if not re.search(r'\b(skills|technical|proficient|expert)\b', text.lower()):
        recommendations.append("List your technical skills with proficiency levels")
    if not re.search(r'\b(projects|portfolio|showcase)\b', text.lower()):
        recommendations.append("Include relevant projects to demonstrate your expertise")
    if not re.search(r'\b(certifications|training|courses)\b', text.lower()):
        recommendations.append("Add relevant certifications and professional training")
    if not re.search(r'\b(languages|bilingual|multilingual)\b', text.lower()):
        recommendations.append("List any additional languages you speak")
    if not re.search(r'\b(publications|research|papers)\b', text.lower()):
        recommendations.append("Include any publications or research papers")
    if not re.search(r'\b(volunteer|community|service)\b', text.lower()):
        recommendations.append("Add volunteer work or community involvement")
    if not re.search(r'\b(achievements|awards|recognition)\b', text.lower()):
        recommendations.append("Highlight your achievements and awards")
    if not re.search(r'\b(interests|hobbies|activities)\b', text.lower()):
        recommendations.append("Consider adding relevant interests or hobbies")
    if not re.search(r'\b(references|recommendations)\b', text.lower()):
        recommendations.append("Include professional references if applicable")

    # Calculate overall score
    overall_score = int((content_score + format_score + style_score + skills_score) / 4)

    # Get ATS verdict
    if overall_score >= 85:
        ats_verdict = "Excellent ATS Compatibility"
    elif overall_score >= 70:
        ats_verdict = "Good ATS Compatibility"
    elif overall_score >= 50:
        ats_verdict = "Average ATS Compatibility"
    else:
        ats_verdict = "Needs Improvement"

    # Get ATS summary
    ats_summary = f"Your resume has {overall_score}% ATS compatibility. {ats_verdict}"

    return {
        'ats_score': overall_score,
        'ats_verdict': ats_verdict,
        'ats_summary': ats_summary,
        'keywords_match': int(content_score),
        'format_compatibility': int(format_score),
        'readability_score': int(style_score),
        'keyword_suggestions': [
            "Include industry-specific keywords from the job description",
            "Add relevant technical skills and certifications",
            "Use standard job titles and industry terms",
            "Include quantifiable achievements and metrics"
        ],
        'format_suggestions': [
            "Use a clean, professional font (Arial, Calibri, or Times New Roman)",
            "Maintain consistent formatting throughout",
            "Use bullet points for better readability",
            "Keep sections clearly organized and labeled"
        ],
        'content_suggestions': [
            "Quantify your achievements with specific numbers and metrics",
            "Use action verbs to start bullet points",
            "Focus on relevant experience and skills",
            "Include a strong professional summary"
        ],
        'technical_skills': [
            {'name': skill, 'matched': skill in text.lower()} 
            for skill in technical_skills
        ],
        'soft_skills': [
            'Communication', 'Leadership', 'Problem Solving', 
            'Teamwork', 'Time Management', 'Adaptability'
        ],
        'recommendations': recommendations[:9]  # Limit to top 9 recommendations
    }

@app.route('/analyze_resume', methods=['POST'])
def analyze_resume():
    try:
        if 'resume' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
            
        file = request.files['resume']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
            
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Only PDF and DOCX files are allowed.'
            }), 400
            
        # Ensure uploads directory exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text based on file type
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(filepath)
        else:
            text = extract_text_from_docx(filepath)
        
        # Analyze the resume
        analysis_result = analyze_resume_content(text)
        
        # Clean up the temporary file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            **analysis_result
        })
        
    except Exception as e:
        logger.error(f"Error analyzing resume: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while analyzing the resume'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 