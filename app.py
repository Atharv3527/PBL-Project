from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.utils
import json
import os
import logging
from datetime import datetime
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'

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

# Load student data from JSON file
def load_student_data():
    try:
        with open('student_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"students": [], "statistics": {
            "total_students": 0,
            "average_performance": 0,
            "top_performance": 0,
            "average_study_hours": 0,
            "last_updated": datetime.now().strftime("%Y-%m-%d")
        }}

# Save student data to JSON file
def save_student_data(data):
    with open('student_data.json', 'w') as f:
        json.dump(data, f, indent=4)

def calculate_statistics(students):
    if not students:
        return {
            "total_students": 0,
            "average_performance": 0,
            "top_performance": 0,
            "average_study_hours": 0,
            "last_updated": datetime.now().strftime("%Y-%m-%d")
        }

    performances = [s['performance'] for s in students]
    study_hours = [s['study_hours'] for s in students]
    
    return {
        "total_students": len(students),
        "average_performance": sum(performances) / len(performances),
        "top_performance": max(performances),
        "average_study_hours": sum(study_hours) / len(study_hours),
        "last_updated": datetime.now().strftime("%Y-%m-%d")
    }

def calculate_dashboard_statistics():
    try:
        with open('student_data.json', 'r') as file:
            students = json.load(file)
            
        total_students = len(students)
        
        if total_students == 0:
            return {
                'total_students': 0,
                'average_performance': 0,
                'top_performance': 0,
                'average_study_hours': 0
            }
        
        # Calculate statistics
        performances = [float(student.get('performance', 0)) for student in students]
        study_hours = [float(student.get('study_hours', 0)) for student in students]
        
        avg_performance = sum(performances) / len(performances)
        top_performance = max(performances)
        avg_study_hours = sum(study_hours) / len(study_hours)
        
        return {
            'total_students': total_students,
            'average_performance': round(avg_performance, 1),
            'top_performance': round(top_performance, 1),
            'average_study_hours': round(avg_study_hours, 1)
        }
    except Exception as e:
        print(f"Error calculating dashboard statistics: {str(e)}")
        return {
            'total_students': 0,
            'average_performance': 0,
            'top_performance': 0,
            'average_study_hours': 0
        }

# Routes
@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    statistics = calculate_statistics()
    return render_template('dashboard.html', statistics=statistics)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/resume_analysis')
def resume_analysis():
    return render_template('resume_analysis.html')

@app.route('/student_data')
def student_data():
    return render_template('student_data.html')

@app.route('/get_statistics')
def get_statistics():
    data = load_student_data()
    return jsonify(data['statistics'])

@app.route('/get_student_data')
def get_student_data():
    try:
        data = load_student_data()
        return jsonify({
            'success': True,
            'data': data['students']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/predict_student/<int:student_id>')
def predict_student(student_id):
    try:
        students = load_student_data()
        if student_id < 0 or student_id >= len(students['students']):
            return jsonify({
                'success': False,
                'error': 'Invalid student ID'
            }), 404
            
        student = students['students'][student_id]
        prediction_result = predict_performance(student)
        
        response = {
            'success': True,
            'student': student['name'],
            'prediction': prediction_result['prediction'],
            'analysis': prediction_result['analysis'],
            'recommendations': get_enhanced_recommendations(student)
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in predict_student: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/delete_student/<int:student_id>', methods=['DELETE'])
def delete_student(student_id):
    try:
        students = load_student_data()
        if student_id < 0 or student_id >= len(students['students']):
            return jsonify({
                'success': False,
                'error': 'Invalid student ID'
            }), 404
            
        deleted_student = students['students'].pop(student_id)
        save_student_data(students)
        
        logger.info(f"Successfully deleted student: {deleted_student['name']}")
        return jsonify({
            'success': True,
            'message': f"Student {deleted_student['name']} deleted successfully"
        })
    except Exception as e:
        logger.error(f"Error in delete_student: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/update_student/<int:student_id>', methods=['PUT'])
def update_student(student_id):
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
            
        # Validate input data
        validate_student_data(data)
        
        students = load_student_data()
        if student_id < 0 or student_id >= len(students['students']):
            return jsonify({
                'success': False,
                'error': 'Invalid student ID'
            }), 404
            
        # Update student data
        students['students'][student_id].update(data)
        
        # Recalculate prediction
        prediction_result = predict_performance(students['students'][student_id])
        students['students'][student_id]['performance'] = prediction_result['prediction']
        
        save_student_data(students)
        
        logger.info(f"Successfully updated student: {students['students'][student_id]['name']}")
        return jsonify({
            'success': True,
            'message': f"Student {students['students'][student_id]['name']} updated successfully",
            'data': students['students'][student_id]
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error in update_student: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'GET':
        return render_template('add_student.html')
    
    try:
        data = load_student_data()
        new_student = request.get_json()
        
        # Add ID and timestamp
        new_student['id'] = len(data['students']) + 1
        new_student['created_at'] = datetime.now().strftime("%Y-%m-%d")
        
        # Add the new student
        data['students'].append(new_student)
        
        # Update statistics
        data['statistics'] = calculate_statistics(data['students'])
        
        # Save updated data
        save_student_data(data)
        
        return jsonify({
            'success': True,
            'message': 'Student added successfully',
            'statistics': data['statistics']
        })
    except Exception as e:
        logger.error(f"Error in add_student: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def normalize_value(value, min_val, max_val):
    """Normalize a value to percentage."""
    return round(((value - min_val) / (max_val - min_val)) * 100)

def get_risk_level(prediction):
    """Determine risk level based on prediction."""
    if prediction >= 80:
        return "Low"
    elif prediction >= 60:
        return "Moderate"
    else:
        return "High"

def get_attendance_impact(attendance):
    """Generate attendance impact message."""
    if attendance >= 90:
        return "Excellent attendance pattern detected"
    elif attendance >= 75:
        return "Good attendance pattern detected"
    else:
        return "Low attendance may affect outcomes"

def get_impact_level(value, threshold):
    """Determine impact level of a feature."""
    if value >= threshold:
        return "Positive"
    elif value >= threshold * 0.8:
        return "Moderate"
    else:
        return "Needs Improvement"

def get_recommendations(data):
    """Generate personalized recommendations based on student data."""
    recommendations = []
    
    if data['study_hours'] < 6:
        recommendations.append({
            'title': 'Increase Study Hours',
            'description': 'Aim for at least 6 hours of focused study time per day'
        })
    
    if data['attendance'] < 85:
        recommendations.append({
            'title': 'Improve Attendance',
            'description': 'Try to maintain at least 85% attendance for better academic performance'
        })
    
    if data['participation_score'] < 7:
        recommendations.append({
            'title': 'Enhance Class Participation',
            'description': 'Actively participate in class discussions and activities'
        })
    
    if data['sleep_duration'] < 7:
        recommendations.append({
            'title': 'Improve Sleep Schedule',
            'description': 'Aim for 7-8 hours of sleep per night for better concentration'
        })
    
    if data['stress_level'] > 7:
        recommendations.append({
            'title': 'Stress Management',
            'description': 'Consider stress-relief activities and time management techniques'
        })
    
    return recommendations

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Process the student data
        student_data = {
            "name": data.get('student_name'),
            "study_hours": float(data.get('study_hours', 0)),
            "attendance": float(data.get('attendance', 0)),
            "previous_grades": float(data.get('previous_grades', 0)),
            "participation_score": float(data.get('participation_score', 0)),
            "socio_economic_status": data.get('socio_economic_status', ''),
            "extracurricular": data.get('extracurricular_activities', ''),
            "learning_style": data.get('learning_style', ''),
            "gender": data.get('gender', ''),
            "parents_education": data.get('parents_education', '')
        }

        # Calculate prediction and analysis
        prediction_result = predict_performance(student_data)
        
        # Prepare response with detailed analysis
        response = {
            'score': prediction_result['prediction'],
            'category': get_risk_level(prediction_result['prediction']),
            'attendance_impact': get_attendance_impact(student_data['attendance']),
            'metrics': {
                'Study Hours': normalize_value(student_data['study_hours'], 0, 24),
                'Attendance': student_data['attendance'],
                'Previous Grades': student_data['previous_grades'],
                'Participation': normalize_value(student_data['participation_score'], 1, 10)
            },
            'feature_importance': [
                {'feature': 'Previous Grades', 'importance': 0.35},
                {'feature': 'Study Hours', 'importance': 0.25},
                {'feature': 'Attendance', 'importance': 0.20},
                {'feature': 'Participation', 'importance': 0.15},
                {'feature': 'Learning Environment', 'importance': 0.05}
            ],
            'analysis': [
                {
                    'feature': 'Study Hours',
                    'value': f"{student_data['study_hours']} hrs/week",
                    'normalized': f"{normalize_value(student_data['study_hours'], 0, 24)}%",
                    'impact': get_impact_level(student_data['study_hours'], 15)
                },
                {
                    'feature': 'Attendance',
                    'value': f"{student_data['attendance']}%",
                    'normalized': f"{student_data['attendance']}%",
                    'impact': get_impact_level(student_data['attendance'], 85)
                },
                {
                    'feature': 'Previous Grades',
                    'value': f"{student_data['previous_grades']}%",
                    'normalized': f"{student_data['previous_grades']}%",
                    'impact': get_impact_level(student_data['previous_grades'], 70)
                },
                {
                    'feature': 'Participation',
                    'value': f"{student_data['participation_score']}/10",
                    'normalized': f"{normalize_value(student_data['participation_score'], 1, 10)}%",
                    'impact': get_impact_level(student_data['participation_score'], 7)
                }
            ],
            'suggestions': get_enhanced_recommendations(student_data)
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_enhanced_recommendations(data):
    """Generate more detailed personalized recommendations based on student data."""
    recommendations = []
    
    if data['study_hours'] < 15:
        hours_needed = max(15 - data['study_hours'], 3)
        recommendations.append(f"Increase study hours by {hours_needed} hours per week for better results")
    
    if data['attendance'] < 85:
        recommendations.append(f"Improve attendance from {data['attendance']}% to at least 85% to ensure better academic outcomes")
    
    if data['participation_score'] < 7:
        recommendations.append("Enhance classroom participation by asking questions and contributing to discussions")
    
    # Learning style specific recommendations
    if data['learning_style'] == 'visual':
        recommendations.append("Use diagrams, charts, and visual aids to support your visual learning style")
    elif data['learning_style'] == 'auditory':
        recommendations.append("Record lectures and use verbal repetition techniques to support your auditory learning style")
    elif data['learning_style'] == 'kinesthetic':
        recommendations.append("Incorporate hands-on activities and practical applications to support your kinesthetic learning style")
    
    # Default recommendations if list is too short
    if len(recommendations) < 3:
        recommendations.append("Form study groups with peers to enhance collaborative learning")
        recommendations.append("Develop a structured study plan with specific goals for each session")
    
    return recommendations

@app.route('/visualize')
def visualize():
    if not load_student_data()['students']:
        return jsonify({'plot': json.dumps({'data': [], 'layout': {}})})
    
    # Create visualization data
    names = [student['name'] for student in load_student_data()['students']]
    performances = [student.get('performance', 0) for student in load_student_data()['students']]
    
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

def predict_performance(student_data):
    try:
        # Convert categorical values to numeric
        timeliness_map = {
            'always on time': 1.0,
            'sometimes late': 0.7,
            'often late': 0.4,
            'never': 0.0
        }
        
        # Prepare features for prediction
        features = {
            'study_hours': float(student_data['study_hours']),
            'attendance': float(student_data['attendance']) / 100,  # Normalize to 0-1
            'previous_grades': float(student_data['previous_grades']) / 100,  # Normalize to 0-1
            'participation_score': float(student_data['participation_score']) / 10,  # Normalize to 0-1
            'timeliness': timeliness_map.get(student_data.get('assignment_submission_timeliness', 'sometimes late'), 0.7)
        }
        
        # Calculate weighted prediction
        weights = {
            'study_hours': 0.25,
            'attendance': 0.2,
            'previous_grades': 0.3,
            'participation_score': 0.15,
            'timeliness': 0.1
        }
        
        prediction = sum(features[key] * weights[key] for key in weights) * 100
        
        # Generate detailed analysis
        analysis = {
            'study_impact': analyze_feature('Study Hours', features['study_hours'], 0.6),
            'attendance_impact': analyze_feature('Attendance', features['attendance'], 0.8),
            'grades_impact': analyze_feature('Previous Grades', features['previous_grades'], 0.7),
            'participation_impact': analyze_feature('Participation', features['participation_score'], 0.7),
            'timeliness_impact': analyze_feature('Assignment Timeliness', features['timeliness'], 0.8)
        }
        
        return {
            'prediction': round(prediction, 2),
            'analysis': analysis,
            'features': features,
            'weights': weights
        }
    except Exception as e:
        logger.error(f"Error in predict_performance: {str(e)}")
        raise

def analyze_feature(name, value, threshold):
    if value >= threshold:
        return {
            'status': 'good',
            'message': f'{name} is at a good level',
            'value': value
        }
    else:
        return {
            'status': 'needs_improvement',
            'message': f'{name} needs improvement',
            'value': value
        }

@app.route('/visualize_performance/<int:student_id>')
def visualize_performance(student_id):
    try:
        students = load_student_data()
        if student_id < 0 or student_id >= len(students['students']):
            return jsonify({
                'success': False,
                'error': 'Invalid student ID'
            }), 404
            
        student = students['students'][student_id]
        prediction_result = predict_performance(student)
        
        # Create radar chart data
        categories = list(prediction_result['features'].keys())
        values = [prediction_result['features'][cat] * 100 for cat in categories]
        
        fig = px.line_polar(
            r=values,
            theta=categories,
            line_close=True,
            range_r=[0, 100],
            title=f"Performance Analysis for {student['name']}"
        )
        
        # Update layout for better visibility
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False
        )
        
        # Convert to JSON for frontend
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            'success': True,
            'chart_data': chart_json,
            'prediction': prediction_result['prediction'],
            'analysis': prediction_result['analysis']
        })
    except Exception as e:
        logger.error(f"Error in visualize_performance: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Add an API endpoint to get updated statistics
@app.route('/api/dashboard-stats')
def get_dashboard_stats():
    return jsonify(calculate_dashboard_statistics())

def analyze_resume_ats(resume_text):
    """Analyze resume for ATS compatibility and provide suggestions."""
    # Common ATS keywords for different fields
    common_keywords = {
        'technical': ['python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'node.js', 'git', 'docker'],
        'soft_skills': ['communication', 'leadership', 'teamwork', 'problem-solving', 'time management'],
        'education': ['bachelor', 'master', 'phd', 'degree', 'university', 'college'],
        'experience': ['years', 'experience', 'responsibilities', 'achievements', 'projects']
    }

    # Calculate keyword match percentage
    keyword_matches = sum(1 for keyword in common_keywords['technical'] if keyword.lower() in resume_text.lower())
    keyword_match_percentage = (keyword_matches / len(common_keywords['technical'])) * 100

    # Check format compatibility
    format_score = 0
    format_issues = []
    
    # Check for common formatting issues
    if len(resume_text.split('\n')) > 2:  # Has multiple sections
        format_score += 25
    else:
        format_issues.append("Consider adding more sections to your resume")
    
    if len(resume_text) > 500:  # Has sufficient content
        format_score += 25
    else:
        format_issues.append("Your resume seems too short")
    
    if any(char.isdigit() for char in resume_text):  # Has numbers/achievements
        format_score += 25
    else:
        format_issues.append("Add quantifiable achievements")
    
    if any(word in resume_text.lower() for word in ['achievement', 'accomplishment', 'result']):
        format_score += 25
    else:
        format_issues.append("Include more achievement statements")

    # Calculate readability score
    words = resume_text.split()
    sentences = resume_text.split('.')
    avg_word_length = sum(len(word) for word in words) / len(words)
    avg_sentence_length = len(words) / len(sentences)
    
    readability_score = 100 - ((avg_word_length - 4) * 10 + (avg_sentence_length - 15) * 2)
    readability_score = max(0, min(100, readability_score))

    # Generate suggestions
    keyword_suggestions = []
    if keyword_match_percentage < 70:
        missing_keywords = [k for k in common_keywords['technical'] if k.lower() not in resume_text.lower()]
        keyword_suggestions.append(f"Add these technical keywords: {', '.join(missing_keywords[:5])}")

    format_suggestions = format_issues if format_issues else ["Your resume format looks good!"]

    content_suggestions = []
    if readability_score < 70:
        content_suggestions.append("Simplify complex sentences")
        content_suggestions.append("Use more bullet points for better readability")
    if not any(word in resume_text.lower() for word in common_keywords['soft_skills']):
        content_suggestions.append("Add more soft skills to your resume")

    # Calculate overall ATS score
    ats_score = (keyword_match_percentage * 0.4 + format_score * 0.3 + readability_score * 0.3)
    
    return {
        'ats_score': round(ats_score),
        'keywords_match': round(keyword_match_percentage),
        'format_compatibility': format_score,
        'readability_score': round(readability_score),
        'keyword_suggestions': keyword_suggestions,
        'format_suggestions': format_suggestions,
        'content_suggestions': content_suggestions,
        'ats_verdict': 'Good ATS Compatibility' if ats_score >= 70 else 'Needs Improvement',
        'ats_summary': 'Your resume has good ATS compatibility' if ats_score >= 70 else 'Consider implementing the suggestions to improve ATS compatibility'
    }

@app.route('/analyze_resume', methods=['POST'])
def analyze_resume():
    try:
        if 'resume' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['resume']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        # Save the file temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(temp_path)

        # Read the resume content
        resume_text = ""
        if file.filename.endswith('.pdf'):
            resume_text = extract_text_from_pdf(temp_path)
        elif file.filename.endswith(('.doc', '.docx')):
            resume_text = extract_text_from_doc(temp_path)

        # Perform ATS analysis
        ats_analysis = analyze_resume_ats(resume_text)

        # Clean up the temporary file
        os.remove(temp_path)

        # Return the analysis results
        return jsonify({
            'success': True,
            **ats_analysis,
            'technical_skills': [
                {'name': 'Python', 'matched': 'python' in resume_text.lower()},
                {'name': 'JavaScript', 'matched': 'javascript' in resume_text.lower()},
                {'name': 'SQL', 'matched': 'sql' in resume_text.lower()},
                {'name': 'HTML/CSS', 'matched': any(x in resume_text.lower() for x in ['html', 'css'])},
                {'name': 'Git', 'matched': 'git' in resume_text.lower()}
            ],
            'skills_match_percentage': ats_analysis['keywords_match'],
            'soft_skills': ['Communication', 'Teamwork', 'Problem Solving', 'Time Management'],
            'education': [
                {'degree': 'Bachelor of Technology', 'institution': 'Example University', 'year': '2020-2024'}
            ],
            'experience': [
                {'position': 'Software Developer', 'company': 'Tech Corp', 'duration': '2022-Present'}
            ],
            'recommendations': [
                'Add more quantifiable achievements',
                'Include relevant project experience',
                'Highlight key technical skills',
                'Use action verbs in descriptions'
            ]
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def extract_text_from_doc(doc_path):
    """Extract text from DOC/DOCX file."""
    try:
        import docx2txt
        return docx2txt.process(doc_path)
    except Exception as e:
        raise Exception(f"Error extracting text from DOC/DOCX: {str(e)}")

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if __name__ == '__main__':
    app.run(debug=True) 