import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, RotateCcw, Volume2 } from 'lucide-react';
import { useEEGData } from '../../hooks/useEEGData';

interface MeditationGuideProps {
  onComplete: (score: number, performance: any) => void;
  onUpdateScore: (points: number) => void;
}

const MeditationGuide: React.FC<MeditationGuideProps> = ({ onComplete, onUpdateScore }) => {
  const [isActive, setIsActive] = useState(false);
  const [phase, setPhase] = useState<'inhale' | 'hold' | 'exhale'>('inhale');
  const [cycleCount, setCycleCount] = useState(0);
  const [timeLeft, setTimeLeft] = useState(300); // 5 minutes
  const [phaseTime, setPhaseTime] = useState(4);
  const [meditationScores, setMeditationScores] = useState<number[]>([]);
  const { currentData } = useEEGData();

  const phaseDurations = {
    inhale: 4,
    hold: 4,
    exhale: 6
  };

  useEffect(() => {
    if (isActive && timeLeft > 0) {
      const timer = setTimeout(() => setTimeLeft(t => t - 1), 1000);
      return () => clearTimeout(timer);
    } else if (timeLeft === 0) {
      // Session complete
      const avgMeditation = meditationScores.length > 0 
        ? meditationScores.reduce((a, b) => a + b, 0) / meditationScores.length 
        : 0;
      
      const consistency = meditationScores.length > 1 
        ? 100 - (Math.max(...meditationScores) - Math.min(...meditationScores)) * 100
        : 100;

      onComplete(Math.round(avgMeditation * 100), {
        accuracy: avgMeditation * 100,
        reactionTime: 0,
        consistency: Math.max(0, consistency),
        focusLevel: avgMeditation * 100
      });
    }
  }, [isActive, timeLeft, meditationScores, onComplete]);

  useEffect(() => {
    if (isActive) {
      const phaseTimer = setTimeout(() => {
        setPhaseTime(t => {
          if (t <= 1) {
            // Move to next phase
            if (phase === 'inhale') {
              setPhase('hold');
              return phaseDurations.hold;
            } else if (phase === 'hold') {
              setPhase('exhale');
              return phaseDurations.exhale;
            } else {
              setPhase('inhale');
              setCycleCount(c => c + 1);
              
              // Record meditation score for this cycle
              const meditationLevel = currentData?.meditation || 0;
              setMeditationScores(prev => [...prev, meditationLevel]);
              onUpdateScore(Math.round(meditationLevel * 10));
              
              return phaseDurations.inhale;
            }
          }
          return t - 1;
        });
      }, 1000);

      return () => clearTimeout(phaseTimer);
    }
  }, [isActive, phase, phaseTime, currentData, onUpdateScore]);

  const getPhaseColor = () => {
    switch (phase) {
      case 'inhale': return 'from-blue-400 to-blue-600';
      case 'hold': return 'from-purple-400 to-purple-600';
      case 'exhale': return 'from-green-400 to-green-600';
    }
  };

  const getPhaseInstruction = () => {
    switch (phase) {
      case 'inhale': return 'Breathe In';
      case 'hold': return 'Hold';
      case 'exhale': return 'Breathe Out';
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const meditationLevel = currentData?.meditation || 0;
  const currentScore = meditationScores.length > 0 
    ? Math.round((meditationScores.reduce((a, b) => a + b, 0) / meditationScores.length) * 100)
    : Math.round(meditationLevel * 100);

  return (
    <div className="relative w-full h-96 bg-gradient-to-br from-indigo-50 to-purple-100 dark:from-indigo-900/20 dark:to-purple-900/30 rounded-xl border border-indigo-200 dark:border-indigo-800 overflow-hidden">
      {/* Header */}
      <div className="absolute top-4 left-4 right-4 flex items-center justify-between z-10">
        <div className="flex items-center space-x-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg px-3 py-1 shadow-sm">
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              Cycles: {cycleCount}
            </span>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg px-3 py-1 shadow-sm">
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              Score: {currentScore}
            </span>
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg px-3 py-1 shadow-sm">
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {formatTime(timeLeft)}
          </span>
        </div>
      </div>

      {/* Meditation Level */}
      <div className="absolute top-16 left-4 bg-white dark:bg-gray-800 rounded-lg p-2 shadow-sm">
        <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">Meditation</div>
        <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <motion.div
            className="bg-purple-500 h-2 rounded-full"
            animate={{ width: `${meditationLevel * 100}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </div>

      {/* Breathing Circle */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative">
          <motion.div
            className={`w-48 h-48 rounded-full bg-gradient-to-br ${getPhaseColor()} shadow-2xl flex items-center justify-center`}
            animate={{
              scale: phase === 'inhale' ? 1.2 : phase === 'hold' ? 1.2 : 0.8,
            }}
            transition={{
              duration: phase === 'inhale' ? 4 : phase === 'hold' ? 0.5 : 6,
              ease: "easeInOut"
            }}
          >
            <div className="text-center text-white">
              <div className="text-2xl font-bold mb-2">
                {getPhaseInstruction()}
              </div>
              <div className="text-lg">
                {phaseTime}
              </div>
            </div>
          </motion.div>
          
          {/* Ripple effect */}
          <motion.div
            className="absolute inset-0 rounded-full border-4 border-white opacity-30"
            animate={{
              scale: [1, 1.5, 1],
              opacity: [0.3, 0, 0.3]
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        </div>
      </div>

      {/* Controls */}
      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2">
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setIsActive(!isActive)}
            className={`p-3 rounded-full shadow-lg transition-colors ${
              isActive 
                ? 'bg-orange-500 hover:bg-orange-600 text-white' 
                : 'bg-green-500 hover:bg-green-600 text-white'
            }`}
          >
            {isActive ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
          </button>
          <button
            onClick={() => {
              setIsActive(false);
              setPhase('inhale');
              setCycleCount(0);
              setPhaseTime(4);
              setTimeLeft(300);
              setMeditationScores([]);
            }}
            className="p-3 bg-gray-500 hover:bg-gray-600 text-white rounded-full shadow-lg transition-colors"
          >
            <RotateCcw className="h-5 w-5" />
          </button>
          <button className="p-3 bg-blue-500 hover:bg-blue-600 text-white rounded-full shadow-lg transition-colors">
            <Volume2 className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* Real-time Feedback */}
      <div className="absolute bottom-4 left-4 right-4 mb-16">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
            Meditation Feedback
          </div>
          <div className="text-sm font-medium text-gray-900 dark:text-white">
            {meditationLevel > 0.7 
              ? "Deep meditation state achieved! Excellent relaxation." 
              : meditationLevel > 0.5 
              ? "Good meditation level. Try to relax deeper." 
              : "Focus on your breathing and let go of thoughts."}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MeditationGuide;