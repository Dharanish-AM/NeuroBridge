import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, Square, Timer, Volume2, VolumeX, Brain, Target } from 'lucide-react';
import { useEEGData } from '../hooks/useEEGData';
import BrainwaveChart from '../components/Charts/BrainwaveChart';

const FocusMode: React.FC = () => {
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [sessionTime, setSessionTime] = useState(0);
  const [targetDuration, setTargetDuration] = useState(25 * 60); // 25 minutes default
  const [isSoundEnabled, setIsSoundEnabled] = useState(true);
  const [breakTime, setBreakTime] = useState(false);
  const { data, currentData, mentalState } = useEEGData();

  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isSessionActive && !breakTime) {
      interval = setInterval(() => {
        setSessionTime(prev => {
          if (prev >= targetDuration) {
            setIsSessionActive(false);
            setBreakTime(true);
            return 0;
          }
          return prev + 1;
        });
      }, 1000);
    }

    return () => clearInterval(interval);
  }, [isSessionActive, breakTime, targetDuration]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const progressPercentage = (sessionTime / targetDuration) * 100;

  const handleStartSession = () => {
    setIsSessionActive(true);
    setBreakTime(false);
    setSessionTime(0);
  };

  const handlePauseSession = () => {
    setIsSessionActive(!isSessionActive);
  };

  const handleStopSession = () => {
    setIsSessionActive(false);
    setBreakTime(false);
    setSessionTime(0);
  };

  const ambientSounds = [
    { name: 'Rain', active: true },
    { name: 'Forest', active: false },
    { name: 'Ocean', active: false },
    { name: 'White Noise', active: false }
  ];

  const focusQuality = currentData ? Math.round(currentData.attention * 100) : 0;
  const sessionScore = Math.round((focusQuality + (sessionTime / targetDuration) * 100) / 2);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Focus Mode
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Deep work sessions with real-time brainwave monitoring
        </p>
      </div>

      {/* Session Status */}
      <div className="max-w-2xl mx-auto">
        <motion.div
          className={`rounded-2xl p-8 text-center ${
            isSessionActive
              ? 'bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-blue-900/20 dark:to-indigo-900/30 border border-blue-200 dark:border-blue-800'
              : 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700'
          }`}
          animate={{
            scale: isSessionActive ? [1, 1.02, 1] : 1,
          }}
          transition={{
            duration: 2,
            repeat: isSessionActive ? Infinity : 0,
            repeatType: "reverse"
          }}
        >
          {/* Timer Display */}
          <div className="mb-8">
            <div className="relative w-48 h-48 mx-auto mb-6">
              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 144 144">
                <circle
                  cx="72"
                  cy="72"
                  r="64"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="text-gray-200 dark:text-gray-700"
                />
                <motion.circle
                  cx="72"
                  cy="72"
                  r="64"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="4"
                  strokeLinecap="round"
                  className="text-blue-500"
                  strokeDasharray={`${2 * Math.PI * 64}`}
                  initial={{ strokeDashoffset: 2 * Math.PI * 64 }}
                  animate={{ 
                    strokeDashoffset: 2 * Math.PI * 64 * (1 - progressPercentage / 100)
                  }}
                  transition={{ duration: 0.5 }}
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <div>
                  <div className="text-4xl font-bold text-gray-900 dark:text-white">
                    {formatTime(Math.max(0, targetDuration - sessionTime))}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    {isSessionActive ? 'Focus Time' : 'Ready to Focus'}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center justify-center space-x-4 mb-6">
            {!isSessionActive ? (
              <motion.button
                onClick={handleStartSession}
                className="flex items-center space-x-2 bg-blue-600 text-white px-8 py-3 rounded-xl font-medium hover:bg-blue-700 transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Play className="h-5 w-5" />
                <span>Start Focus Session</span>
              </motion.button>
            ) : (
              <div className="flex space-x-3">
                <motion.button
                  onClick={handlePauseSession}
                  className="flex items-center space-x-2 bg-orange-500 text-white px-6 py-3 rounded-xl font-medium hover:bg-orange-600 transition-colors"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Pause className="h-5 w-5" />
                  <span>Pause</span>
                </motion.button>
                <motion.button
                  onClick={handleStopSession}
                  className="flex items-center space-x-2 bg-red-500 text-white px-6 py-3 rounded-xl font-medium hover:bg-red-600 transition-colors"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Square className="h-5 w-5" />
                  <span>Stop</span>
                </motion.button>
              </div>
            )}
          </div>

          {/* Session Settings */}
          <div className="flex items-center justify-center space-x-6 text-sm text-gray-600 dark:text-gray-400">
            <div className="flex items-center space-x-2">
              <Timer className="h-4 w-4" />
              <select
                value={targetDuration}
                onChange={(e) => setTargetDuration(Number(e.target.value))}
                disabled={isSessionActive}
                className="bg-transparent border border-gray-300 dark:border-gray-600 rounded px-2 py-1"
              >
                <option value={15 * 60}>15 min</option>
                <option value={25 * 60}>25 min</option>
                <option value={45 * 60}>45 min</option>
                <option value={60 * 60}>60 min</option>
              </select>
            </div>
            <button
              onClick={() => setIsSoundEnabled(!isSoundEnabled)}
              className="flex items-center space-x-2 hover:text-gray-900 dark:hover:text-white transition-colors"
            >
              {isSoundEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
              <span>Ambient</span>
            </button>
          </div>
        </motion.div>
      </div>

      {/* Real-time Feedback */}
      {isSessionActive && (
        <motion.div
          className="grid grid-cols-1 lg:grid-cols-3 gap-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {/* Focus Quality */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center space-x-3 mb-4">
              <Target className="h-5 w-5 text-blue-500" />
              <h3 className="font-semibold text-gray-900 dark:text-white">Focus Quality</h3>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                {focusQuality}%
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <motion.div
                  className={`h-2 rounded-full ${
                    focusQuality >= 70 ? 'bg-green-500' :
                    focusQuality >= 50 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  animate={{ width: `${focusQuality}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                {focusQuality >= 70 ? 'Excellent focus!' :
                 focusQuality >= 50 ? 'Good focus' : 'Try to concentrate'}
              </p>
            </div>
          </div>

          {/* Mental State */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center space-x-3 mb-4">
              <Brain className="h-5 w-5 text-purple-500" />
              <h3 className="font-semibold text-gray-900 dark:text-white">Mental State</h3>
            </div>
            <div className="text-center">
              <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${mentalState.color} text-white mb-3`}>
                {mentalState.primary}
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                {mentalState.description}
              </p>
            </div>
          </div>

          {/* Session Score */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center space-x-3 mb-4">
              <Timer className="h-5 w-5 text-green-500" />
              <h3 className="font-semibold text-gray-900 dark:text-white">Session Score</h3>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                {sessionScore}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Time: {formatTime(sessionTime)}
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Live Brainwave Chart */}
      {isSessionActive && data.length > 0 && (
        <motion.div
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
            Live Brainwave Activity
          </h3>
          <BrainwaveChart data={data} height={250} />
        </motion.div>
      )}

      {/* Ambient Sounds */}
      {isSoundEnabled && (
        <motion.div
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Ambient Sounds
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {ambientSounds.map((sound) => (
              <button
                key={sound.name}
                className={`p-3 rounded-lg text-sm font-medium transition-colors ${
                  sound.active
                    ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400'
                    : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                {sound.name}
              </button>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default FocusMode;