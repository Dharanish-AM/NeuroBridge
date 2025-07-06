import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Target, CheckCircle, X, Timer } from 'lucide-react';
import { useEEGData } from '../../hooks/useEEGData';

interface FocusEnhancerProps {
  onComplete: (score: number, performance: any) => void;
  onUpdateScore: (points: number) => void;
}

const FocusEnhancer: React.FC<FocusEnhancerProps> = ({ onComplete, onUpdateScore }) => {
  const [currentTarget, setCurrentTarget] = useState<{ x: number; y: number; id: number } | null>(null);
  const [score, setScore] = useState(0);
  const [round, setRound] = useState(1);
  const [timeLeft, setTimeLeft] = useState(60);
  const [hits, setHits] = useState(0);
  const [misses, setMisses] = useState(0);
  const [reactionTimes, setReactionTimes] = useState<number[]>([]);
  const [targetStartTime, setTargetStartTime] = useState(0);
  const { currentData } = useEEGData();

  const generateTarget = useCallback(() => {
    const x = Math.random() * 80 + 10; // 10-90% of container width
    const y = Math.random() * 80 + 10; // 10-90% of container height
    const id = Date.now();
    
    setCurrentTarget({ x, y, id });
    setTargetStartTime(Date.now());
    
    // Auto-miss after 2 seconds
    setTimeout(() => {
      setCurrentTarget(prev => {
        if (prev && prev.id === id) {
          setMisses(m => m + 1);
          return null;
        }
        return prev;
      });
    }, 2000);
  }, []);

  const handleTargetHit = useCallback(() => {
    if (currentTarget && targetStartTime) {
      const reactionTime = Date.now() - targetStartTime;
      setReactionTimes(prev => [...prev, reactionTime]);
      
      // Score based on reaction time and focus level
      const focusBonus = (currentData?.attention || 0.5) * 20;
      const speedBonus = Math.max(0, 30 - (reactionTime / 100));
      const points = Math.round(10 + focusBonus + speedBonus);
      
      setScore(s => s + points);
      setHits(h => h + 1);
      onUpdateScore(points);
      
      setCurrentTarget(null);
      
      // Generate next target after short delay
      setTimeout(generateTarget, 500 + Math.random() * 1000);
    }
  }, [currentTarget, targetStartTime, currentData, onUpdateScore, generateTarget]);

  useEffect(() => {
    generateTarget();
  }, [generateTarget]);

  useEffect(() => {
    if (timeLeft > 0) {
      const timer = setTimeout(() => setTimeLeft(t => t - 1), 1000);
      return () => clearTimeout(timer);
    } else {
      // Training complete
      const avgReactionTime = reactionTimes.length > 0 
        ? reactionTimes.reduce((a, b) => a + b, 0) / reactionTimes.length 
        : 0;
      
      const accuracy = hits + misses > 0 ? (hits / (hits + misses)) * 100 : 0;
      const avgFocus = currentData?.attention || 0;
      
      onComplete(score, {
        accuracy,
        reactionTime: avgReactionTime,
        consistency: Math.max(0, 100 - (Math.max(...reactionTimes) - Math.min(...reactionTimes)) / 10),
        focusLevel: avgFocus * 100
      });
    }
  }, [timeLeft, score, hits, misses, reactionTimes, currentData, onComplete]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="relative w-full h-96 bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-blue-900/20 dark:to-indigo-900/30 rounded-xl border border-blue-200 dark:border-blue-800 overflow-hidden">
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
              Hits: {hits} | Misses: {misses}
            </span>
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg px-3 py-1 shadow-sm flex items-center space-x-2">
          <Timer className="h-4 w-4 text-blue-500" />
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {formatTime(timeLeft)}
          </span>
        </div>
      </div>

      {/* Focus Level Indicator */}
      <div className="absolute top-16 left-4 bg-white dark:bg-gray-800 rounded-lg p-2 shadow-sm">
        <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">Focus</div>
        <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <motion.div
            className="bg-blue-500 h-2 rounded-full"
            animate={{ width: `${(currentData?.attention || 0) * 100}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </div>

      {/* Game Area */}
      <div className="absolute inset-0 pt-20">
        <AnimatePresence>
          {currentTarget && (
            <motion.button
              key={currentTarget.id}
              className="absolute w-16 h-16 bg-red-500 rounded-full shadow-lg flex items-center justify-center hover:bg-red-600 transition-colors"
              style={{
                left: `${currentTarget.x}%`,
                top: `${currentTarget.y}%`,
                transform: 'translate(-50%, -50%)'
              }}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0, opacity: 0 }}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={handleTargetHit}
            >
              <Target className="h-8 w-8 text-white" />
            </motion.button>
          )}
        </AnimatePresence>

        {/* Instructions */}
        {!currentTarget && timeLeft > 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="text-lg font-medium text-gray-600 dark:text-gray-400 mb-2">
                Get ready for the next target...
              </div>
              <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
            </div>
          </div>
        )}
      </div>

      {/* Real-time Feedback */}
      <div className="absolute bottom-4 left-4 right-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
            Real-time Coaching
          </div>
          <div className="text-sm font-medium text-gray-900 dark:text-white">
            {(currentData?.attention || 0) > 0.7 
              ? "Excellent focus! Keep it up!" 
              : (currentData?.attention || 0) > 0.5 
              ? "Good concentration. Try to maintain it." 
              : "Focus is dropping. Take a deep breath and concentrate."}
          </div>
        </div>
      </div>
    </div>
  );
};

export default FocusEnhancer;