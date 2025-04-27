import React, { useState } from 'react';
import { Form, Button, Card, Alert, Row, Col, ListGroup } from 'react-bootstrap';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const Predict = () => {
  const [formData, setFormData] = useState({
    study_hours: '',
    attendance: '',
    previous_grades: '',
    participation_score: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setPrediction(null);
    setSuggestions([]);

    try {
      // Validate form data
      const numericFields = ['study_hours', 'attendance', 'previous_grades', 'participation_score'];
      for (const field of numericFields) {
        const value = parseFloat(formData[field]);
        if (isNaN(value)) {
          throw new Error(`${field.replace('_', ' ')} must be a number`);
        }
      }

      // Convert string values to numbers
      const predictionData = {
        study_hours: parseFloat(formData.study_hours),
        attendance: parseFloat(formData.attendance),
        previous_grades: parseFloat(formData.previous_grades),
        participation_score: parseFloat(formData.participation_score)
      };

      // Send data to the server
      const response = await axios.post('http://localhost:5000/predict', predictionData);
      
      setPrediction(response.data.prediction);
      setSuggestions(response.data.suggestions);
      
    } catch (err) {
      setError(err.message || 'An error occurred during prediction.');
    } finally {
      setLoading(false);
    }
  };

  const getPredictionColorClass = (value) => {
    if (value >= 80) return 'text-success';
    if (value >= 60) return 'text-primary';
    if (value >= 40) return 'text-warning';
    return 'text-danger';
  };

  return (
    <div style={{ backgroundColor: '#e6f9e6', minHeight: '100vh', padding: '2rem' }}>
      <h1 className="text-center mb-4">Predict Student Performance</h1>
      
      <Row>
        <Col md={6}>
          <Card className="shadow-sm mb-4">
            <Card.Body>
              <Card.Title>Student Information</Card.Title>
              
              {error && <Alert variant="danger">{error}</Alert>}
              
              <Form onSubmit={handleSubmit}>
                <Form.Group className="mb-3">
                  <Form.Label>Study Hours (per day)</Form.Label>
                  <Form.Control
                    type="number"
                    step="0.1"
                    min="0"
                    max="24"
                    name="study_hours"
                    value={formData.study_hours}
                    onChange={handleChange}
                    placeholder="Enter average study hours per day"
                    required
                  />
                  <Form.Text className="text-muted">
                    Enter a value between 0 and 24
                  </Form.Text>
                </Form.Group>
                
                <Form.Group className="mb-3">
                  <Form.Label>Attendance (%)</Form.Label>
                  <Form.Control
                    type="number"
                    step="0.1"
                    min="0"
                    max="100"
                    name="attendance"
                    value={formData.attendance}
                    onChange={handleChange}
                    placeholder="Enter attendance percentage"
                    required
                  />
                  <Form.Text className="text-muted">
                    Enter a value between 0 and 100
                  </Form.Text>
                </Form.Group>
                
                <Form.Group className="mb-3">
                  <Form.Label>Previous Grades</Form.Label>
                  <Form.Control
                    type="number"
                    step="0.1"
                    min="0"
                    max="100"
                    name="previous_grades"
                    value={formData.previous_grades}
                    onChange={handleChange}
                    placeholder="Enter previous grades (0-100)"
                    required
                  />
                  <Form.Text className="text-muted">
                    Enter a value between 0 and 100
                  </Form.Text>
                </Form.Group>
                
                <Form.Group className="mb-3">
                  <Form.Label>Participation Score</Form.Label>
                  <Form.Control
                    type="number"
                    step="0.1"
                    min="0"
                    max="10"
                    name="participation_score"
                    value={formData.participation_score}
                    onChange={handleChange}
                    placeholder="Enter participation score (0-10)"
                    required
                  />
                  <Form.Text className="text-muted">
                    Enter a value between 0 and 10
                  </Form.Text>
                </Form.Group>
                
                <div className="d-flex justify-content-end gap-2 mt-4">
                  <Button variant="secondary" onClick={() => setFormData({ study_hours: '', attendance: '', previous_grades: '', participation_score: '' })}>
                    Reset
                  </Button>
                  <Button variant="contained" color="secondary" onClick={() => navigate('/student-data')}>
                    Go to Student Data
                  </Button>
                  <Button variant="primary" type="submit" disabled={loading}>
                    {loading ? 'Predicting...' : 'Predict Performance'}
                  </Button>
                </div>
              </Form>
            </Card.Body>
          </Card>
        </Col>
        
        <Col md={6}>
          {prediction !== null && (
            <Card className="shadow-sm mb-4 h-100">
              <Card.Body>
                <Card.Title>Prediction Results</Card.Title>
                
                <div className="text-center my-4">
                  <h2>Predicted Performance</h2>
                  <h1 className={`display-1 fw-bold ${getPredictionColorClass(prediction)}`}>
                    {prediction}%
                  </h1>
                  <p className="text-muted">
                    {prediction >= 80 ? 'Excellent' : 
                     prediction >= 60 ? 'Good' : 
                     prediction >= 40 ? 'Average' : 'Needs Improvement'}
                  </p>
                </div>
                
                <h4 className="mt-4">Suggestions for Improvement</h4>
                <ListGroup variant="flush">
                  {suggestions.map((suggestion, index) => (
                    <ListGroup.Item key={index} className="border-0 ps-0">
                      <i className="bi bi-check-circle-fill text-success me-2"></i>
                      {suggestion}
                    </ListGroup.Item>
                  ))}
                </ListGroup>
              </Card.Body>
            </Card>
          )}
          
          {!prediction && !loading && (
            <Card className="shadow-sm mb-4 h-100 bg-light">
              <Card.Body className="d-flex flex-column justify-content-center align-items-center text-center">
                <i className="bi bi-graph-up display-1 text-muted mb-3"></i>
                <h3 className="text-muted">Enter student data and click "Predict Performance"</h3>
                <p className="text-muted">
                  The prediction model will analyze the data and provide a performance prediction along with suggestions for improvement.
                </p>
              </Card.Body>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  );
};

export default Predict; 