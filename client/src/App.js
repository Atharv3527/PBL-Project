import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Predict from './pages/Predict';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Predict />} />
        <Route path="/predict" element={<Predict />} />
      </Routes>
    </Router>
  );
}

export default App; 