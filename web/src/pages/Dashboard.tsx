import React from 'react';
import { motion } from 'framer-motion';
import { Activity, Brain, Zap, Wifi, WifiOff, Target, Timer } from 'lucide-react';
import { useEEGData } from '../hooks/useEEGData';
import BrainwaveChart from '../components/Charts/BrainwaveChart';
import RadarChart from '../components/Charts/RadarChart';
import StatCard from '../components/UI/StatCard';
import MentalStateCard from '../components/UI/MentalStateCard';

const Dashboard: React.FC = () => {
  const { data, currentData, isConnected, mentalState, toggleConnection } = useEEGData();

  const stats = currentData ? [
    {
      title: 'Attention Level',
      value: `${Math.round(currentData.attention * 100)}%`,
      icon: Target,
      color: 'bg-blue-500',
      trend: 'up' as const,
      trendValue: '+5% from yesterday'
    },
    {
      title: 'Meditation State',
      value: `${Math.round(currentData.meditation * 100)}%`,
      icon: Brain,
      color: 'bg-green-500',
      trend: 'stable' as const,
      trendValue: 'Steady for 10 minutes'
    },
    {
      title: 'Signal Quality',
      value: `${Math.round(currentData.quality * 100)}%`,
      icon: Activity,
      color: 'bg-purple-500',
      trend: currentData.quality > 0.8 ? 'up' : 'down',
      trendValue: currentData.quality > 0.8 ? 'Excellent' : 'Good'
    },
    {
      title: 'Session Time',
      value: `${Math.floor(data.length / 10)}s`,
      icon: Timer,
      color: 'bg-orange-500',
      subtitle: `${data.length} samples recorded`
    }
  ] : [];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Live EEG Monitor
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Real-time brainwave analysis and mental state monitoring
          </p>
        </div>
        <motion.button
          onClick={toggleConnection}
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
            isConnected
              ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
              : 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
          }`}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {isConnected ? (
            <>
              <Wifi className="h-4 w-4" />
              <span>Connected</span>
            </>
          ) : (
            <>
              <WifiOff className="h-4 w-4" />
              <span>Disconnected</span>
            </>
          )}
        </motion.button>
      </div>

      {/* Mental State Card */}
      <MentalStateCard mentalState={mentalState} />

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
          >
            <StatCard {...stat} />
          </motion.div>
        ))}
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Real-time Brainwave Chart */}
        <motion.div
          className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Real-time Brainwaves
            </h3>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-600 dark:text-gray-400">Live</span>
            </div>
          </div>
          <BrainwaveChart data={data} height={400} />
        </motion.div>

        {/* Brainwave Distribution */}
        <motion.div
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
            Current Distribution
          </h3>
          <RadarChart data={currentData} height={350} />
        </motion.div>
      </div>

      {/* Live Tips */}
      <motion.div
        className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <div className="flex items-start space-x-4">
          <div className="p-2 bg-blue-500 rounded-lg">
            <Zap className="h-5 w-5 text-white" />
          </div>
          <div>
            <h4 className="font-semibold text-blue-900 dark:text-blue-100">
              Live Coaching Tip
            </h4>
            <p className="text-blue-700 dark:text-blue-300 mt-1">
              {mentalState.primary === 'Focused' 
                ? "Great focus! Try to maintain this state by keeping your posture straight and taking deep breaths."
                : mentalState.primary === 'Relaxed'
                ? "Perfect for creative thinking! This is an ideal state for brainstorming and problem-solving."
                : mentalState.primary === 'Drowsy'
                ? "Consider taking a short break, hydrating, or doing some light stretches to re-energize."
                : "Try some deep breathing exercises to help regulate your mental state."
              }
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Dashboard;