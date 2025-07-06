import { useState, useEffect, useCallback } from 'react';

export const useEEGData = () => {
  const [data, setData] = useState([]);
  const [currentData, setCurrentData] = useState(null);
  const [isConnected, setIsConnected] = useState(true);
  const [mentalState, setMentalState] = useState({
    primary: 'Focused',
    secondary: 'Alert',
    confidence: 85,
    description: 'Your mind is in an optimal state for learning and concentration.',
    color: 'bg-blue-500'
  });

  const generateEEGReading = useCallback(() => {
    const baseTime = Date.now();
    const noise = () => (Math.random() - 0.5) * 0.1;
    
    // Simulate realistic EEG patterns
    const time = baseTime / 1000;
    const alpha = Math.max(0, 0.6 + 0.3 * Math.sin(time * 0.1) + noise());
    const beta = Math.max(0, 0.4 + 0.2 * Math.cos(time * 0.15) + noise());
    const gamma = Math.max(0, 0.3 + 0.1 * Math.sin(time * 0.2) + noise());
    const delta = Math.max(0, 0.8 + 0.2 * Math.cos(time * 0.05) + noise());
    const theta = Math.max(0, 0.5 + 0.25 * Math.sin(time * 0.12) + noise());
    
    const attention = Math.max(0, Math.min(1, (alpha + beta) / 2 + noise()));
    const meditation = Math.max(0, Math.min(1, (alpha + theta) / 2 + noise()));
    const quality = Math.max(0.7, Math.min(1, 0.9 + noise()));

    return {
      timestamp: baseTime,
      alpha,
      beta,
      gamma,
      delta,
      theta,
      attention,
      meditation,
      quality
    };
  }, []);

  const updateMentalState = useCallback((eegData) => {
    const { alpha, beta, theta, attention, meditation } = eegData;
    
    let primary = 'Neutral';
    let secondary = 'Baseline';
    let description = 'Your brainwave patterns are in a normal resting state.';
    let color = 'bg-gray-500';
    let confidence = 70;

    if (attention > 0.7 && beta > 0.5) {
      primary = 'Focused';
      secondary = 'Alert';
      description = 'Your mind is in an optimal state for learning and concentration.';
      color = 'bg-blue-500';
      confidence = Math.round(attention * 100);
    } else if (meditation > 0.6 && alpha > 0.6) {
      primary = 'Relaxed';
      secondary = 'Calm';
      description = 'You are in a peaceful, meditative state. Great for creativity.';
      color = 'bg-green-500';
      confidence = Math.round(meditation * 100);
    } else if (theta > 0.6) {
      primary = 'Drowsy';
      secondary = 'Tired';
      description = 'Your brain shows signs of fatigue. Consider taking a break.';
      color = 'bg-orange-500';
      confidence = Math.round((1 - theta) * 100);
    } else if (beta > 0.7) {
      primary = 'Stressed';
      secondary = 'Anxious';
      description = 'High beta activity detected. Try some deep breathing exercises.';
      color = 'bg-red-500';
      confidence = Math.round(beta * 100);
    }

    setMentalState({ primary, secondary, confidence, description, color });
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      const newReading = generateEEGReading();
      setCurrentData(newReading);
      setData(prev => {
        const updated = [...prev, newReading];
        return updated.slice(-100); // Keep last 100 readings
      });
      updateMentalState(newReading);
    }, 100); // 10Hz sampling rate

    return () => clearInterval(interval);
  }, [generateEEGReading, updateMentalState]);

  const toggleConnection = () => {
    setIsConnected(!isConnected);
  };

  return {
    data,
    currentData,
    isConnected,
    mentalState,
    toggleConnection
  };
};