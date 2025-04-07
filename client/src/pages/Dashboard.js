import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Table } from 'react-bootstrap';
import { Bar, Pie } from 'react-chartjs-2';
import axios from 'axios';
import {
  Chart as ChartJS,
  ArcElement,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  ArcElement,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const Dashboard = () => {
  const [students, setStudents] = useState([]);
  const [stats, setStats] = useState({
    average_performance: 0,
    total_students: 0,
    performance_distribution: { labels: [], data: [] }
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const studentsResponse = await axios.get('http://localhost:5000/api/students');
        const statsResponse = await axios.get('http://localhost:5000/api/stats');
        
        setStudents(studentsResponse.data);
        setStats(statsResponse.data);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

  // Chart data for performance distribution
  const pieChartData = {
    labels: stats.performance_distribution.labels || [],
    datasets: [
      {
        data: stats.performance_distribution.data || [],
        backgroundColor: [
          'rgba(255, 99, 132, 0.6)',
          'rgba(54, 162, 235, 0.6)',
          'rgba(255, 206, 86, 0.6)',
          'rgba(75, 192, 192, 0.6)',
          'rgba(153, 102, 255, 0.6)',
        ],
        borderWidth: 1,
      },
    ],
  };

  // Bar chart for study hours
  const barChartData = {
    labels: students.map(student => student.name || 'Unnamed'),
    datasets: [
      {
        label: 'Study Hours',
        data: students.map(student => student.study_hours || 0),
        backgroundColor: 'rgba(54, 162, 235, 0.6)',
      },
    ],
  };

  return (
    <div>
      <h1 className="text-center mb-4">Student Performance Dashboard</h1>
      
      {/* Stats Cards */}
      <Row className="mb-4">
        <Col md={6} lg={3} className="mb-3">
          <Card className="text-center h-100 shadow-sm">
            <Card.Body>
              <Card.Title>Total Students</Card.Title>
              <h3>{stats.total_students}</h3>
            </Card.Body>
          </Card>
        </Col>
        <Col md={6} lg={3} className="mb-3">
          <Card className="text-center h-100 shadow-sm">
            <Card.Body>
              <Card.Title>Average Performance</Card.Title>
              <h3>{stats.average_performance}%</h3>
            </Card.Body>
          </Card>
        </Col>
        <Col md={6} lg={3} className="mb-3">
          <Card className="text-center h-100 shadow-sm">
            <Card.Body>
              <Card.Title>Top Performance</Card.Title>
              <h3>
                {students.length > 0
                  ? Math.max(...students.map(s => s.performance || 0))
                  : 0}%
              </h3>
            </Card.Body>
          </Card>
        </Col>
        <Col md={6} lg={3} className="mb-3">
          <Card className="text-center h-100 shadow-sm">
            <Card.Body>
              <Card.Title>Average Study Hours</Card.Title>
              <h3>
                {students.length > 0
                  ? (students.reduce((acc, student) => acc + (student.study_hours || 0), 0) / students.length).toFixed(1)
                  : 0}
              </h3>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Charts */}
      <Row className="mb-4">
        <Col md={6} className="mb-3">
          <Card className="h-100 shadow-sm">
            <Card.Body>
              <Card.Title>Performance Distribution</Card.Title>
              <div className="chart-container">
                <Pie data={pieChartData} options={{ maintainAspectRatio: false }} />
              </div>
            </Card.Body>
          </Card>
        </Col>
        <Col md={6} className="mb-3">
          <Card className="h-100 shadow-sm">
            <Card.Body>
              <Card.Title>Study Hours by Student</Card.Title>
              <div className="chart-container">
                <Bar 
                  data={barChartData} 
                  options={{ 
                    maintainAspectRatio: false,
                    scales: {
                      y: {
                        beginAtZero: true,
                        title: {
                          display: true,
                          text: 'Hours'
                        }
                      }
                    }
                  }} 
                />
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Students Table */}
      <Card className="shadow-sm">
        <Card.Body>
          <Card.Title>Student Data</Card.Title>
          <div className="table-responsive">
            <Table striped bordered hover>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Name</th>
                  <th>Study Hours</th>
                  <th>Attendance (%)</th>
                  <th>Previous Grades</th>
                  <th>Participation</th>
                  <th>Performance</th>
                </tr>
              </thead>
              <tbody>
                {students.length > 0 ? (
                  students.map((student, index) => (
                    <tr key={index}>
                      <td>{index + 1}</td>
                      <td>{student.name}</td>
                      <td>{student.study_hours}</td>
                      <td>{student.attendance}</td>
                      <td>{student.previous_grades}</td>
                      <td>{student.participation_score}</td>
                      <td>{student.performance || 'N/A'}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="7" className="text-center">No students data available</td>
                  </tr>
                )}
              </tbody>
            </Table>
          </div>
        </Card.Body>
      </Card>
    </div>
  );
};

export default Dashboard; 