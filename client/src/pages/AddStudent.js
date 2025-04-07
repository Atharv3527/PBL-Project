import React, { useState } from 'react';
import { Form, Button, Card, Alert } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const AddStudent = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: '',
    study_hours: '',
    attendance: '',
    previous_grades: '',
    participation_score: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

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
    setSuccess('');

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
      const studentData = {
        ...formData,
        study_hours: parseFloat(formData.study_hours),
        attendance: parseFloat(formData.attendance),
        previous_grades: parseFloat(formData.previous_grades),
        participation_score: parseFloat(formData.participation_score)
      };

      // Send data to the server
      const response = await axios.post('http://localhost:5000/api/students', studentData);
      
      setSuccess('Student added successfully!');
      
      // Reset form after successful submission
      setFormData({
        name: '',
        study_hours: '',
        attendance: '',
        previous_grades: '',
        participation_score: ''
      });
      
      // Redirect to dashboard after a short delay
      setTimeout(() => {
        navigate('/');
      }, 2000);
      
    } catch (err) {
      setError(err.message || 'An error occurred while adding the student.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1 className="text-center mb-4">Add New Student</h1>
      
      <Card className="mx-auto shadow-sm" style={{ maxWidth: '800px' }}>
        <Card.Body>
          <Card.Title>Student Information</Card.Title>
          
          {error && <Alert variant="danger">{error}</Alert>}
          {success && <Alert variant="success">{success}</Alert>}
          
          <Form onSubmit={handleSubmit}>
            <Form.Group className="mb-3">
              <Form.Label>Student Name</Form.Label>
              <Form.Control
                type="text"
                name="name"
                value={formData.name}
                onChange={handleChange}
                placeholder="Enter student name"
                required
              />
            </Form.Group>
            
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
            
            <div className="d-grid gap-2 d-md-flex justify-content-md-end">
              <Button 
                variant="secondary" 
                onClick={() => navigate('/')}
                className="me-md-2"
              >
                Cancel
              </Button>
              <Button 
                variant="primary" 
                type="submit"
                disabled={loading}
              >
                {loading ? 'Submitting...' : 'Add Student'}
              </Button>
            </div>
          </Form>
        </Card.Body>
      </Card>
    </div>
  );
};

export default AddStudent; 