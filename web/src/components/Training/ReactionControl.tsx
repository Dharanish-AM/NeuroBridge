import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Zap, X, CheckCircle, AlertTriangle } from 'lucide-react';
import { useEEGData } from '../../hooks/useEEGData';

interface ReactionControlProps {
  onComplete: (score: number, performance: any) => void;
  onUpdateScore: (points: number) => void;
}

const ReactionControl: React.FC<ReactionControlProps> = ({ onComplete, onUpdateScore }) => {
  const [currentStimulus, setCurrentStimulus] = useState<{
    type: 'go' | 'stop' | 'wait';
    color: string;
    startTime: number;
    id: number;
  } | null>(null);
  const [score, setScore] = useState(0);
  const [round, setRound] = useState(1);
  const [timeLeft, setTimeLeft] = useState(90); // 1.5 minutes
  const [correctResponses, setCorrectResponses] = useState(0);
  const [incorrectResponses, setIncorrectResponses] = useState(0);
  const [reactionTimes, setReactionTimes] = useState<number[]>([]);
  const [inhibitionSuccess, setInhibitionSuccess] = useState(0);
  const [inhibitionFails, setInhibitionFails] = useState(0);
  const { currentData } = useEEGData();

  const generateStimulus = useCallback(() => {
    const stimulusTypes = ['go', 'go', 'go', 'stop']; // 75% go, 25% stop
    const type = stimulusTypes[Math.floor(Math.random() * stimulusTypes.length)] as 'go' | 'stop';
    
    const colors = {
      go: 'bg-green-500',
      stop: 'bg-red-500'
    };

    const stimulus = {
      type,
      color: colors[type],
      startTime: Date.now(),
      id: Date.now()
    };

    setCurrentStimulus(stimulus);

    // Auto-timeout after 1.5 seconds
    setTimeout(() => {
      setCurrentStimulus(prev => {
        if (prev && prev.id === stimulus.id) {
          if (prev.type === 'go') {
            // Missed a go signal
            setIncorrectResponses(i => i + 1);
          } else {
            // Successfully inhibited a stop signal
            setInhibitionSuccess(i => i + 1);
            const points = Math.round(15 + (currentData?.attention || 0.5) * 10);
            setScore(s => s + points);
            onUpdateScore(points);
          }
          return null;
        }
        return prev;
      });
    }, 1500);
  }, [currentData, onUpdateScore]);

  const handleResponse = useCallback(() => {
    if (currentStimulus) {
      const reactionTime = Date.now() - currentStimulus.startTime;
      setReactionTimes(prev => [...prev, reactionTime]);

      if (currentStimulus.type === 'go') {
        // Correct response to go signal
        setCorrectResponses(c => c + 1);
        
        // Score based on reaction time and focus level
        const focusBonus = (currentData?.attention || 0.5) * 20;
        const speedBonus = Math.max(0, 25 - (reactionTime / 50));
        const points = Math.round(10 + focusBonus + speedBonus);
        
        setScore(s => s + points);
        onUpdateScore(points);
      } else {
        // Incorrect response to stop signal (failed inhibition)
        setInhibitionFails(i => i + 1);
        setIncorrectResponses(i => i + 1);
      }

      setCurrentStimulus(null);
    }
  }, [currentStimulus, currentData, onUpdateScore]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (!currentStimulus && timeLeft > 0) {
        // Random delay between stimuli (1-3 seconds)
        setTimeout(generateStimulus, 1000 + Math.random() * 2000);
      }
    }, 100);

    return () => clearInterval(interval);
  }, [currentStimulus, timeLeft, generateStimulus]);

  useEffect(() => {
    if (timeLeft > 0) {
      const timer = setTimeout(() => setTimeLeft(t => t - 1), 1000);
      return () => clearTimeout(timer);
    } else {
      // Training complete
      const totalResponses = correctResponses + incorrectResponses;
      const accuracy = totalResponses > 0 ? (correctResponses / totalResponses) * 100 : 0;
      const avgReactionTime = reactionTimes.length > 0 
        ? reactionTimes.reduce((a, b) => a + b, 0) / reactionTimes.length 
        : 0;
      const inhibitionRate = (inhibitionSuccess + inhibitionFails) > 0 
        ? (inhibitionSuccess / (inhibitionSuccess + inhibitionFails)) * 100 
        : 0;

      onComplete(score, {
        accuracy,
        reactionTime: avgReactionTime,
        consistency: inhibitionRate,
        focusLevel: (currentData?.attention || 0) * 100
      });
    }
  }, [timeLeft, score, correctResponses, incorrectResponses, reactionTimes, inhibitionSuccess, inhibitionFails, currentData, onComplete]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const totalResponses = correctResponses + incorrectResponses;
  const accuracy = totalResponses > 0 ? Math.round((correctResponses / totalResponses) * 100) : 0;
  const avgReactionTime = reactionTimes.length > 0 
    ? Math.round(reactionTimes.reduce((a, b) => a + b, 0) / reactionTimes.length)
    : 0;

  return (
    <div className="relative w-full h-96 bg-gradient-to-br from-orange-50 to-red-100 dark:from-orange-900/20 dark:to-red-900/30 rounded-xl border border-orange-200 dark:border-orange-800 overflow-hidden">
      {/* Header */}
      <div className="absolute top-4 left-4 right-4 flex items-center justify-between z-10">
        <div className="flex items-center space-x-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg px-3 py-1 shadow-sm">
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              Score: {score}
            </span>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg px-3 py-1 shadow-sm">
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              Accuracy: {accuracy}%
            </span>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg px-3 py-1 shadow-sm">
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              Avg RT: {avgReactionTime}ms
            </span>
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg px-3 py-1 shadow-sm">
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {formatTime(timeLeft)}
          </span>
        </div>
      </div>

      {/* Focus Level */}
      <div className="absolute top-16 left-4 bg-white dark:bg-gray-800 rounded-lg p-2 shadow-sm">
        <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">Focus</div>
        <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <motion.div
            className="bg-orange-500 h-2 rounded-full"
            animate={{ width: `${(currentData?.attention || 0) * 100}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </div>

      {/* Game Area */}
      <div className="absolute inset-0 pt-20 pb-16">
        <div className="h-full flex items-center justify-center">
          <AnimatePresence>
            {currentStimulus ? (
              <motion.button
                key={currentStimulus.id}
                className={`w-32 h-32 rounded-full shadow-2xl flex items-center justify-center text-white font-bold text-xl ${currentStimulus.color} hover:scale-105 transition-transform`}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0, opacity: 0 }}
                onClick={handleResponse}
                whileTap={{ scale: 0.9 }}
              >
                {currentStimulus.type === 'go' ? (
                  <CheckCircle className="h-16 w-16" />
                ) : (
                  <X className="h-16 w-16" />
                )}
              </motion.button>
            ) : (
              <div className="text-center">
                <div className="w-32 h-32 rounded-full bg-gray-300 dark:bg-gray-600 flex items-center justify-center mb-4">
                  <Zap className="h-16 w-16 text-gray-500" />
                </div>
                <div className="text-lg font-medium text-gray-600 dark:text-gray-400">
                  Get ready...
                </div>
              </div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Instructions */}
      <div className="absolute bottom-4 left-4 right-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm">
          <div className="grid grid-cols-2 gap-4 mb-3">
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-5 w-5 text-green-500" />
              <span className="text-sm text-gray-900 dark:text-white">
                Green: Click quickly!
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <X className="h-5 w-5 text-red-500" />
              <span className="text-sm text-gray-900 dark:text-white">
                Red: Don't click!
              </span>
            </div>
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400">
            Inhibition Success: {inhibitionSuccess} | Fails: {inhibitionFails}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReactionControl;