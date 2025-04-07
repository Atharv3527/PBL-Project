// Global variable to store students data
let studentsData = [];
let chartInstances = {}; // Store chart instances for reuse

// Setup initial state and event listeners when DOM content is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Fetch initial data
    fetchStudents();
    fetchStats();
    
    // Setup form submit handlers
    document.getElementById('add-student-form').addEventListener('submit', handleAddStudent);
    document.getElementById('predict-form').addEventListener('submit', handlePredictPerformance);
    
    // Setup navigation
    setupNavigation();
    
    // Setup save prediction button (handle it only when it's visible)
    document.addEventListener('click', function(e) {
        if (e.target && e.target.id === 'save-prediction-button') {
            saveStudentWithPrediction();
        }
    });
    
    // Setup student data table show/hide functionality
    const studentDataTableCard = document.getElementById('student-data-table-card');
    const showStudentDataBtn = document.getElementById('show-student-data-btn');
    
    // Initially hide the student data table
    studentDataTableCard.style.display = 'none';
    
    // Add click event to toggle the table visibility
    showStudentDataBtn.addEventListener('click', function() {
        if (studentDataTableCard.style.display === 'none') {
            studentDataTableCard.style.display = 'block';
            studentDataTableCard.classList.add('show');
            showStudentDataBtn.innerHTML = '<i class="bi bi-eye-slash me-2"></i>Hide Student Data';
            studentDataTableCard.scrollIntoView({ behavior: 'smooth' });
        } else {
            setTimeout(() => {
                studentDataTableCard.style.display = 'none';
            }, 500); // Match this with the CSS transition time
            studentDataTableCard.classList.remove('show');
            showStudentDataBtn.innerHTML = '<i class="bi bi-table me-2"></i>View Student Data';
        }
    });
});

// Setup navigation between sections
function setupNavigation() {
    // Get all nav links
    const navLinks = document.querySelectorAll('.nav-link');
    
    // Add click event to each link
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Get the target section id from href
            const targetId = this.getAttribute('href').substring(1);
            
            // Show the target section
            showSection(targetId);
            
            // Update active class on nav links
            updateNavigation(targetId);
        });
    });
}

// Update active navigation link
function updateNavigation(sectionId) {
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(l => l.classList.remove('active'));
    document.querySelector(`a[href="#${sectionId}"]`).classList.add('active');
}

// Function to show a specific section and hide others
function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('section').forEach(section => {
        section.classList.add('d-none');
    });
    
    // Show target section
    document.getElementById(sectionId).classList.remove('d-none');
    
    // Scroll to top
    window.scrollTo(0, 0);
}

// Fetch students data from API
function fetchStudents() {
    fetch('/api/students')
        .then(response => response.json())
        .then(data => {
            studentsData = data;
            updateStudentsTable();
        })
        .catch(error => {
            console.error('Error fetching students:', error);
        });
}

// Fetch statistics data from API
function fetchStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            updateStatistics(data);
        })
        .catch(error => {
            console.error('Error fetching stats:', error);
        });
}

// Update students table with data
function updateStudentsTable() {
    const tableBody = document.getElementById('students-table-body');
    
    // Clear table
    tableBody.innerHTML = '';
    
    if (studentsData.length === 0) {
        // Show no data message
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="10" class="text-center">No students data available</td>';
        tableBody.appendChild(row);
    } else {
        // Add rows for each student
        studentsData.forEach((student, index) => {
            const row = document.createElement('tr');
            
            // Format performance with appropriate class
            let performanceHTML = 'N/A';
            if (student.performance) {
                let performanceClass = '';
                if (student.performance >= 80) {
                    performanceClass = 'performance-excellent';
                } else if (student.performance >= 60) {
                    performanceClass = 'performance-good';
                } else if (student.performance >= 40) {
                    performanceClass = 'performance-average';
                } else {
                    performanceClass = 'performance-poor';
                }
                performanceHTML = `<span class="${performanceClass}">${student.performance}%</span>`;
            }
            
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${student.name}</td>
                <td>${student.study_hours}</td>
                <td>${student.attendance}</td>
                <td>${student.previous_grades}</td>
                <td>${student.participation_score}</td>
                <td>${student.socio_economic_status || 'N/A'}</td>
                <td>${performanceHTML}</td>
                <td>
                    <div class="btn-group">
                        <button class="btn btn-sm btn-predict" onclick="predictForStudent(${index})">
                            <i class="bi bi-lightning-fill me-1"></i> Predict
                        </button>
                        <button class="btn btn-sm btn-delete" onclick="deleteStudent(${index})">
                            <i class="bi bi-trash-fill me-1"></i> Delete
                        </button>
                    </div>
                </td>
            `;
            
            tableBody.appendChild(row);
        });
    }
}

// Function to delete a student
function deleteStudent(studentIndex) {
    if (confirm('Are you sure you want to delete this student?')) {
        const studentToDelete = studentsData[studentIndex];
        
        // Send delete request to API
        fetch(`/api/students/${studentIndex}`, {
            method: 'DELETE'
        })
        .then(response => response.json())
        .then(data => {
            // Remove student from array
            studentsData.splice(studentIndex, 1);
            
            // Update table
            updateStudentsTable();
            
            // Update stats
            fetchStats();
            
            // Show a temporary success message
            const dashboardSection = document.getElementById('dashboard');
            
            // Create alert element
            const alertEl = document.createElement('div');
            alertEl.className = 'alert alert-success alert-dismissible fade show';
            alertEl.setAttribute('role', 'alert');
            alertEl.innerHTML = `
                Student deleted successfully!
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            `;
            
            // Insert after heading
            dashboardSection.querySelector('h1').after(alertEl);
            
            // Remove after 3 seconds
            setTimeout(() => {
                alertEl.remove();
            }, 3000);
        })
        .catch(error => {
            console.error('Error deleting student:', error);
            alert('Failed to delete student. Please try again.');
        });
    }
}

// Function to handle predict button click for a specific student
function predictForStudent(studentIndex) {
    const student = studentsData[studentIndex];
    
    // Show predict section
    showSection('predict');
    updateNavigation('predict');
    
    // Fill form with student data
    if (student.name) document.getElementById('predict_name').value = student.name;
    document.getElementById('predict_study_hours').value = student.study_hours;
    document.getElementById('predict_attendance').value = student.attendance;
    document.getElementById('predict_previous_grades').value = student.previous_grades;
    document.getElementById('predict_participation_score').value = student.participation_score;
    
    // Set selects if they exist
    if (student.socio_economic_status) {
        document.getElementById('predict_socio_economic_status').value = student.socio_economic_status;
    }
    if (student.extracurricular) {
        document.getElementById('predict_extracurricular').value = student.extracurricular;
    }
    if (student.learning_style) {
        document.getElementById('predict_learning_style').value = student.learning_style;
    }
    if (student.gender) {
        document.getElementById('predict_gender').value = student.gender;
    }
    if (student.parents_education) {
        document.getElementById('predict_parents_education').value = student.parents_education;
    }
    if (student.study_environment) {
        document.getElementById('predict_study_environment').value = student.study_environment;
    }
    
    // Set new fields if they exist
    if (student.parent_meeting_freq) {
        document.getElementById('predict_parent_meeting_freq').value = student.parent_meeting_freq;
    }
    if (student.home_support) {
        document.getElementById('predict_home_support').value = student.home_support;
    }
    if (student.sleep_duration) {
        document.getElementById('predict_sleep_duration').value = student.sleep_duration;
    }
    if (student.stress_level) {
        document.getElementById('predict_stress_level').value = student.stress_level;
    }
    if (student.physical_activity) {
        document.getElementById('predict_physical_activity').value = student.physical_activity;
    }
    if (student.peer_group_quality) {
        document.getElementById('predict_peer_group_quality').value = student.peer_group_quality;
    }
    if (student.submission_timeliness) {
        document.getElementById('predict_submission_timeliness').value = student.submission_timeliness;
    }
    
    // Scroll to prediction form
    document.getElementById('predict').scrollIntoView({ behavior: 'smooth' });
}

// Update statistics display
function updateStatistics(data) {
    document.getElementById('total-students').textContent = data.total_students;
    document.getElementById('avg-performance').textContent = `${data.average_performance}%`;
    
    // Calculate top performance
    let topPerformance = 0;
    if (studentsData.length > 0) {
        topPerformance = Math.max(...studentsData.map(s => s.performance || 0));
    }
    document.getElementById('top-performance').textContent = `${topPerformance}%`;
    
    // Calculate average study hours
    let avgStudyHours = 0;
    if (studentsData.length > 0) {
        avgStudyHours = studentsData.reduce((acc, s) => acc + (s.study_hours || 0), 0) / studentsData.length;
    }
    document.getElementById('avg-study-hours').textContent = avgStudyHours.toFixed(1);
}

// Handle add student form submission
function handleAddStudent(e) {
    e.preventDefault();
    
    // Get form data
    const formData = {
        name: document.getElementById('name').value,
        study_hours: parseFloat(document.getElementById('study_hours').value),
        attendance: parseFloat(document.getElementById('attendance').value),
        previous_grades: parseFloat(document.getElementById('previous_grades').value),
        participation_score: parseFloat(document.getElementById('participation_score').value),
        socio_economic_status: document.getElementById('socio_economic_status').value,
        extracurricular: document.getElementById('extracurricular').value,
        learning_style: document.getElementById('learning_style').value,
        gender: document.getElementById('gender').value,
        parents_education: document.getElementById('parents_education').value,
        study_environment: document.getElementById('study_environment').value,
        parent_meeting_freq: document.getElementById('parent_meeting_freq').value,
        home_support: document.getElementById('home_support').value,
        sleep_duration: parseFloat(document.getElementById('sleep_duration').value),
        stress_level: document.getElementById('stress_level').value,
        physical_activity: document.getElementById('physical_activity').value,
        peer_group_quality: document.getElementById('peer_group_quality').value,
        submission_timeliness: document.getElementById('submission_timeliness').value
    };
    
    // Show loading state
    const submitBtn = document.getElementById('add-student-submit');
    const originalText = submitBtn.textContent;
    submitBtn.textContent = 'Adding...';
    submitBtn.disabled = true;
    
    // Send data to API
    fetch('/api/students', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        // Show success message
        const alertEl = document.getElementById('add-student-alert');
        alertEl.textContent = 'Student added successfully!';
        alertEl.classList.remove('d-none', 'alert-danger');
        alertEl.classList.add('alert-success');
        
        // Reset form
        document.getElementById('add-student-form').reset();
        
        // Refresh data
        fetchStudents();
        fetchStats();
        
        // Redirect to dashboard after a delay
        setTimeout(() => {
            showSection('dashboard');
            updateNavigation('dashboard');
            alertEl.classList.add('d-none');
        }, 2000);
    })
    .catch(error => {
        // Show error message
        const alertEl = document.getElementById('add-student-alert');
        alertEl.textContent = 'Error adding student. Please try again.';
        alertEl.classList.remove('d-none', 'alert-success');
        alertEl.classList.add('alert-danger');
        
        console.error('Error adding student:', error);
    })
    .finally(() => {
        // Reset button state
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
    });
}

// Handle predict performance form submission
function handlePredictPerformance(e) {
    e.preventDefault();
    
    // Get form data
    const formData = {
        name: document.getElementById('predict_name').value,
        study_hours: parseFloat(document.getElementById('predict_study_hours').value),
        attendance: parseFloat(document.getElementById('predict_attendance').value),
        previous_grades: parseFloat(document.getElementById('predict_previous_grades').value),
        participation_score: parseFloat(document.getElementById('predict_participation_score').value),
        socio_economic_status: document.getElementById('predict_socio_economic_status').value,
        extracurricular: document.getElementById('predict_extracurricular').value,
        learning_style: document.getElementById('predict_learning_style').value,
        gender: document.getElementById('predict_gender').value,
        parents_education: document.getElementById('predict_parents_education').value,
        study_environment: document.getElementById('predict_study_environment').value,
        parent_meeting_freq: document.getElementById('predict_parent_meeting_freq').value,
        home_support: document.getElementById('predict_home_support').value,
        sleep_duration: parseFloat(document.getElementById('predict_sleep_duration').value),
        stress_level: document.getElementById('predict_stress_level').value,
        physical_activity: document.getElementById('predict_physical_activity').value,
        peer_group_quality: document.getElementById('predict_peer_group_quality').value,
        submission_timeliness: document.getElementById('predict_submission_timeliness').value
    };
    
    // Show loading state
    const submitBtn = document.getElementById('predict-submit');
    const originalText = submitBtn.textContent;
    submitBtn.textContent = 'Predicting...';
    submitBtn.disabled = true;
    
    // Send data to API
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        // Update prediction results
        updatePredictionResults(data, formData);
    })
    .catch(error => {
        // Show error message
        const alertEl = document.getElementById('predict-alert');
        alertEl.textContent = 'Error making prediction. Please try again.';
        alertEl.classList.remove('d-none', 'alert-success');
        alertEl.classList.add('alert-danger');
        
        console.error('Error predicting performance:', error);
    })
    .finally(() => {
        // Reset button state
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
    });
}

// Update prediction results display
function updatePredictionResults(data, formData) {
    const resultsCard = document.getElementById('prediction-results-card');
    
    // Determine performance class
    let performanceClass = 'performance-poor';
    let performanceText = 'Needs Improvement';
    
    if (data.prediction >= 80) {
        performanceClass = 'performance-excellent';
        performanceText = 'Excellent';
    } else if (data.prediction >= 60) {
        performanceClass = 'performance-good';
        performanceText = 'Good';
    } else if (data.prediction >= 40) {
        performanceClass = 'performance-average';
        performanceText = 'Average';
    }
    
    // Create a more detailed visualization section
    const visualizationsSection = `
        <div class="row mt-4">
            <div class="col-12">
                <h5 class="mb-3">Performance Analysis</h5>
                <div class="row">
                    <div class="col-md-6 mb-4">
                        <div class="card h-100">
                            <div class="card-body">
                                <h5 class="card-title">Feature Importance</h5>
                                <div id="feature-importance-chart" style="height: 300px;"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6 mb-4">
                        <div class="card h-100">
                            <div class="card-body">
                                <h5 class="card-title">Study Hours Impact</h5>
                                <div id="study-hours-chart" style="height: 300px;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Create factors table for the input data
    const factorsHtml = `
        <table class="table table-sm">
            <thead>
                <tr>
                    <th>Factor</th>
                    <th>Value</th>
                    <th>Impact</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Study Hours</td>
                    <td>${formData.study_hours}</td>
                    <td>${getFactorImpact(formData.study_hours, 0, 24, 'High')}</td>
                </tr>
                <tr>
                    <td>Attendance</td>
                    <td>${formData.attendance}%</td>
                    <td>${getFactorImpact(formData.attendance, 0, 100, 'Very High')}</td>
                </tr>
                <tr>
                    <td>Previous Grades</td>
                    <td>${formData.previous_grades}</td>
                    <td>${getFactorImpact(formData.previous_grades, 0, 100, 'Medium')}</td>
                </tr>
                <tr>
                    <td>Participation</td>
                    <td>${formData.participation_score}/10</td>
                    <td>${getFactorImpact(formData.participation_score, 0, 10, 'Medium')}</td>
                </tr>
                <tr>
                    <td>Sleep Duration</td>
                    <td>${formData.sleep_duration} hrs</td>
                    <td>${getFactorImpact(formData.sleep_duration, 4, 12, 'Medium')}</td>
                </tr>
                <tr>
                    <td>Stress Level</td>
                    <td>${formData.stress_level}</td>
                    <td>Medium</td>
                </tr>
                <tr>
                    <td>Physical Activity</td>
                    <td>${formData.physical_activity}</td>
                    <td>Low</td>
                </tr>
            </tbody>
        </table>
    `;
    
    // Create gauge chart for performance level
    const gaugeChartHTML = `
        <div class="col-md-6 mb-4">
            <div class="card h-100">
                <div class="card-body">
                    <h5 class="card-title">Performance Level</h5>
                    <div id="gauge-chart" style="height: 250px;"></div>
                </div>
            </div>
        </div>
    `;
    
    // Update results card content
    resultsCard.innerHTML = `
        <div class="card-body">
            <h5 class="card-title mb-4">Prediction Results ${formData.name ? 'for ' + formData.name : ''}</h5>
            
            <div class="text-center mb-4">
                <h2>Predicted Performance</h2>
                <h1 class="display-1 fw-bold ${performanceClass}">
                    ${data.prediction}%
                </h1>
                <p class="text-muted">${performanceText}</p>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <h5 class="mb-3">Performance Factors</h5>
                    ${factorsHtml}
                </div>
                
                <div class="col-md-6">
                    <h5 class="mb-3">Improvement Suggestions</h5>
                    <div class="card bg-light mb-3">
                        <div class="card-body">
                            <ul class="list-group list-group-flush">
                                ${data.suggestions.map(suggestion => `
                                    <li class="list-group-item bg-light border-0 ps-0">
                                        <i class="bi bi-check-circle-fill text-success me-2"></i>
                                        ${suggestion}
                                    </li>
                                `).join('')}
                            </ul>
                        </div>
                    </div>
                    
                    <div class="mt-4">
                        <button class="btn btn-primary w-100" onclick="saveStudentWithPrediction()">
                            <i class="bi bi-save me-2"></i> Save Student with Prediction
                        </button>
                    </div>
                </div>
            </div>
            
            ${visualizationsSection}
        </div>
    `;

    // Scroll to the results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Create charts with Plotly
    setTimeout(() => {
        createFeatureImportanceChart(data.feature_importance);
        createStudyHoursChart();
    }, 100);
}

// Create feature importance bar chart using Plotly
function createFeatureImportanceChart(featureImportance) {
    // If no feature importance data is provided, create sample data
    if (!featureImportance) {
        featureImportance = {
            'study_hours': 0.35,
            'attendance': 0.25,
            'previous_grades': 0.15,
            'participation': 0.10,
            'sleep_duration': 0.08,
            'stress_level': 0.04,
            'physical_activity': 0.03
        };
    }
    
    // Convert to arrays for Plotly
    const features = Object.keys(featureImportance);
    const importance = Object.values(featureImportance);
    
    // Sort by importance
    const sortedIndices = importance.map((val, idx) => ({ val, idx }))
        .sort((a, b) => b.val - a.val)
        .map(obj => obj.idx);
    
    const sortedFeatures = sortedIndices.map(idx => features[idx]);
    const sortedImportance = sortedIndices.map(idx => importance[idx]);
    
    // Create colors based on importance
    const colors = sortedImportance.map(value => {
        // Color gradient from red to green
        const r = Math.round(255 * (1 - value));
        const g = Math.round(255 * value);
        return `rgb(${r}, ${g}, 100)`;
    });
    
    // Create the plot
    const data = [{
        type: 'bar',
        x: sortedImportance,
        y: sortedFeatures,
        orientation: 'h',
        marker: {
            color: colors
        }
    }];
    
    const layout = {
        margin: { l: 150, r: 20, t: 10, b: 40 },
        xaxis: {
            title: 'Importance',
            range: [0, Math.max(...sortedImportance) * 1.1]
        },
        yaxis: {
            automargin: true
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {
            family: 'Segoe UI, sans-serif'
        }
    };
    
    Plotly.newPlot('feature-importance-chart', data, layout, {responsive: true});
}

// Create study hours scatter plot with regression line
function createStudyHoursChart() {
    // Sample data points for study hours vs performance
    const studyHours = [2, 3, 4, 5, 6, 7, 8, 9];
    const performance = [50, 58, 65, 72, 78, 85, 90, 95];
    
    // Linear regression line
    const trace1 = {
        x: studyHours,
        y: performance,
        mode: 'markers',
        type: 'scatter',
        name: 'Student Data',
        marker: {
            size: 10,
            color: 'rgba(79, 70, 229, 0.7)'
        }
    };
    
    // Calculate regression line
    const n = studyHours.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    for (let i = 0; i < n; i++) {
        sumX += studyHours[i];
        sumY += performance[i];
        sumXY += studyHours[i] * performance[i];
        sumX2 += studyHours[i] * studyHours[i];
    }
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    const regressionX = [0, 10];
    const regressionY = regressionX.map(x => slope * x + intercept);
    
    const trace2 = {
        x: regressionX,
        y: regressionY,
        mode: 'lines',
        type: 'scatter',
        name: 'Trend Line',
        line: {
            color: 'rgba(220, 53, 69, 0.7)',
            width: 2
        }
    };
    
    const data = [trace1, trace2];
    
    const layout = {
        margin: { l: 50, r: 20, t: 10, b: 40 },
        xaxis: {
            title: 'Study Hours per Day',
            range: [0, 10]
        },
        yaxis: {
            title: 'Performance (%)',
            range: [40, 100]
        },
        legend: {
            orientation: 'h',
            y: -0.2
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {
            family: 'Segoe UI, sans-serif'
        }
    };
    
    Plotly.newPlot('study-hours-chart', data, layout, {responsive: true});
}

// Helper function to determine the impact of a factor
function getFactorImpact(value, min, max, defaultImpact) {
    const range = max - min;
    const percentage = (value - min) / range;
    
    if (percentage < 0.3) return 'Low';
    if (percentage < 0.7) return 'Medium';
    return 'High';
}

// Function to save a student with the predicted performance
function saveStudentWithPrediction() {
    // Get form data
    const formData = {
        name: document.getElementById('predict_name').value || 'Anonymous Student',
        study_hours: parseFloat(document.getElementById('predict_study_hours').value),
        attendance: parseFloat(document.getElementById('predict_attendance').value),
        previous_grades: parseFloat(document.getElementById('predict_previous_grades').value),
        participation_score: parseFloat(document.getElementById('predict_participation_score').value),
        socio_economic_status: document.getElementById('predict_socio_economic_status').value,
        extracurricular: document.getElementById('predict_extracurricular').value,
        learning_style: document.getElementById('predict_learning_style').value,
        gender: document.getElementById('predict_gender').value,
        parents_education: document.getElementById('predict_parents_education').value,
        study_environment: document.getElementById('predict_study_environment').value,
        parent_meeting_freq: document.getElementById('predict_parent_meeting_freq').value,
        home_support: document.getElementById('predict_home_support').value,
        sleep_duration: parseFloat(document.getElementById('predict_sleep_duration').value),
        stress_level: document.getElementById('predict_stress_level').value,
        physical_activity: document.getElementById('predict_physical_activity').value,
        peer_group_quality: document.getElementById('predict_peer_group_quality').value,
        submission_timeliness: document.getElementById('predict_submission_timeliness').value,
        // Get the predicted performance from the display
        performance: parseFloat(document.querySelector('.display-1').textContent)
    };
    
    // Send data to API
    fetch('/api/students', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        // Show alert
        const alertEl = document.getElementById('predict-alert');
        alertEl.textContent = 'Student saved successfully with prediction!';
        alertEl.classList.remove('d-none', 'alert-danger');
        alertEl.classList.add('alert-success');
        
        // Reset form after a delay
        setTimeout(() => {
            document.getElementById('predict-form').reset();
            alertEl.classList.add('d-none');
            
            // Refresh data
            fetchStudents();
            fetchStats();
            
            // Redirect to dashboard
            showSection('dashboard');
            updateNavigation('dashboard');
        }, 2000);
    })
    .catch(error => {
        // Show error message
        const alertEl = document.getElementById('predict-alert');
        alertEl.textContent = 'Error saving student. Please try again.';
        alertEl.classList.remove('d-none', 'alert-success');
        alertEl.classList.add('alert-danger');
        
        console.error('Error saving student:', error);
    });
} 