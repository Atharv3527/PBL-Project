import React, { useState } from 'react';
import { Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';

const PredictPerformance = () => {
  const navigate = useNavigate();

  const handleReset = () => {
    // Implement reset logic here
  };

  const handleSubmit = () => {
    // Implement submit logic here
  };

  return (
    <div>
      <Button variant="outlined" onClick={handleReset} sx={{ mr: 2 }}>
        Reset
      </Button>
      <Button variant="contained" color="primary" onClick={handleSubmit} sx={{ mr: 2 }}>
        Predict Performance
      </Button>
      <Button variant="contained" color="secondary" onClick={() => navigate('/student-data')}>
        Go to Student Data
      </Button>
    </div>
  );
};

export default PredictPerformance; 