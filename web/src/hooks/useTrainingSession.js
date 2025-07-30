import { useState, useEffect, useCallback } from 'react';
import { useEEGData } from './useEEGData';

export const useTrainingSession = () => {
  const [currentSession, setCurrentSession] = useState(null);
  const [sessions, setSessions] = useState([]);
  const { currentData } = useEEGData();

  const startSession = useCallback((moduleId) => {
    const newSession = {
      id: `session_${Date.now()}`,
      moduleId,
      startTime: Date.now(),
      score: 0,
      maxScore: 100,
      duration: 0,
      completed: false,
      performance: {
        accuracy: 0,
        reactionTime: 0,
        consistency: 0,
        focusLevel: currentData?.attention || 0
      }
    };
    setCurrentSession(newSession);
  }, [currentData]);

  const updateScore = useCallback((points) => {
    if (currentSession) {
      setCurrentSession(prev => prev ? {
        ...prev,
        score: Math.min(prev.score + points, prev.maxScore)
      } : null);
    }
  }, [currentSession]);

  const updatePerformance = useCallback((performance) => {
    if (currentSession) {
      setCurrentSession(prev => prev ? {
        ...prev,
        performance: { ...prev.performance, ...performance }
      } : null);
    }
  }, [currentSession]);

  const endSession = useCallback(() => {
    if (currentSession) {
      const endTime = Date.now();
      const completedSession = {
        ...currentSession,
        endTime,
        duration: endTime - currentSession.startTime,
        completed: true,
        performance: {
          ...currentSession.performance,
          focusLevel: currentData?.attention || 0
        }
      };
      
      setSessions(prev => [...prev, completedSession]);
      setCurrentSession(null);
      
      return completedSession;
    }
    return null;
  }, [currentSession, currentData]);

  return {
    currentSession,
    sessions,
    startSession,
    updateScore,
    updatePerformance,
    endSession
  };
};