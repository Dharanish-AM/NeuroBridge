import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Target, Brain, Gamepad2, Timer, Award, Play, X, Trophy } from 'lucide-react';
import { useTrainingSession } from '../hooks/useTrainingSession';
import FocusEnhancer from '../components/Training/FocusEnhancer';
import MeditationGuide from '../components/Training/MeditationGuide';
import MemoryMatch from '../components/Training/MemoryMatch';
import ReactionControl from '../components/Training/ReactionControl';

const Training: React.FC = () => {
  const [activeModule, setActiveModule] = useState<string | null>(null);
  const [completedSessions, setCompletedSessions] = useState<string[]>([]);
  const { currentSession, sessions, startSession, updateScore, endSession } = useTrainingSession();

  const trainingModules = [
    {
      id: 'focus',
      title: 'Focus Enhancer',
      description: 'Improve concentration through targeted attention exercises',
      icon: Target,
      color: 'bg-blue-500',
      difficulty: 'Beginner',
      duration: '10-15 min',
      sessions: sessions.filter(s => s.moduleId === 'focus').length,
      component: FocusEnhancer
    },
    {
      id: 'meditation',
      title: 'Meditation Guide',
      description: 'EEG-guided meditation for deep relaxation and mindfulness',
      icon: Brain,
      color: 'bg-green-500',
      difficulty: 'All Levels',
      duration: '5-20 min',
      sessions: sessions.filter(s => s.moduleId === 'meditation').length,
      component: MeditationGuide
    },
    {
      id: 'memory',
      title: 'Memory Match',
      description: 'Cognitive training games that adapt to your brainwave patterns',
      icon: Gamepad2,
      color: 'bg-purple-500',
      difficulty: 'Intermediate',
      duration: '8-12 min',
      sessions: sessions.filter(s => s.moduleId === 'memory').length,
      component: MemoryMatch
    },
    {
      id: 'reaction',
      title: 'Reaction Control',
      description: 'Train attention inhibition and impulse control',
      icon: Timer,
      color: 'bg-orange-500',
      difficulty: 'Advanced',
      duration: '15-20 min',
      sessions: sessions.filter(s => s.moduleId === 'reaction').length,
      component: ReactionControl
    }
  ];

  const achievements = [
    { 
      title: 'Focus Master', 
      description: '10 focus sessions completed', 
      progress: Math.min(100, (sessions.filter(s => s.moduleId === 'focus').length / 10) * 100),
      unlocked: sessions.filter(s => s.moduleId === 'focus').length >= 10
    },
    { 
      title: 'Zen Mind', 
      description: '50 meditation minutes', 
      progress: Math.min(100, (sessions.filter(s => s.moduleId === 'meditation').reduce((acc, s) => acc + s.duration, 0) / (50 * 60 * 1000)) * 100),
      unlocked: sessions.filter(s => s.moduleId === 'meditation').reduce((acc, s) => acc + s.duration, 0) >= 50 * 60 * 1000
    },
    { 
      title: 'Memory Champion', 
      description: 'Score 95% in memory games', 
      progress: Math.min(100, Math.max(...sessions.filter(s => s.moduleId === 'memory').map(s => s.performance.accuracy), 0)),
      unlocked: sessions.some(s => s.moduleId === 'memory' && s.performance.accuracy >= 95)
    },
    { 
      title: 'Quick Thinker', 
      description: 'React under 200ms consistently', 
      progress: Math.min(100, sessions.filter(s => s.moduleId === 'reaction' && s.performance.reactionTime < 200).length * 10),
      unlocked: sessions.filter(s => s.moduleId === 'reaction' && s.performance.reactionTime < 200).length >= 10
    }
  ];

  const handleStartTraining = (moduleId: string) => {
    setActiveModule(moduleId);
    startSession(moduleId);
  };

  const handleTrainingComplete = (score: number, performance: any) => {
    const completedSession = endSession();
    if (completedSession) {
      setCompletedSessions(prev => [...prev, completedSession.id]);
    }
    setActiveModule(null);
  };

  const handleCloseTraining = () => {
    if (currentSession) {
      endSession();
    }
    setActiveModule(null);
  };

  const activeModuleData = trainingModules.find(m => m.id === activeModule);
  const ActiveComponent = activeModuleData?.component;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Mental Training Modules
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Enhance your cognitive abilities with EEG-guided training programs
        </p>
      </div>

      {/* Active Training Session */}
      <AnimatePresence>
        {activeModule && ActiveComponent && (
          <motion.div
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="bg-white dark:bg-gray-800 rounded-xl p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
            >
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg ${activeModuleData.color}`}>
                    <activeModuleData.icon className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                      {activeModuleData.title}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400">
                      {currentSession && (
                        <span>Session Score: {currentSession.score}</span>
                      )}
                    </p>
                  </div>
                </div>
                <button
                  onClick={handleCloseTraining}
                  className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                >
                  <X className="h-6 w-6" />
                </button>
              </div>

              <ActiveComponent
                onComplete={handleTrainingComplete}
                onUpdateScore={updateScore}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Training Modules Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {trainingModules.map((module, index) => {
          const Icon = module.icon;
          const recentSessions = sessions.filter(s => s.moduleId === module.id).slice(-5);
          const avgScore = recentSessions.length > 0 
            ? Math.round(recentSessions.reduce((acc, s) => acc + s.score, 0) / recentSessions.length)
            : 0;

          return (
            <motion.div
              key={module.id}
              className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              whileHover={{ scale: 1.02, y: -2 }}
            >
              <div className="flex items-start space-x-4">
                <div className={`p-3 rounded-lg ${module.color}`}>
                  <Icon className="h-6 w-6 text-white" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    {module.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300 text-sm mt-1">
                    {module.description}
                  </p>
                  <div className="flex items-center space-x-4 mt-3">
                    <span className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                      {module.difficulty}
                    </span>
                    <span className="text-xs text-gray-500">
                      {module.duration}
                    </span>
                    <span className="text-xs text-gray-500">
                      {module.sessions} sessions
                    </span>
                  </div>
                  
                  {avgScore > 0 && (
                    <div className="mt-3 p-2 bg-gray-50 dark:bg-gray-700 rounded">
                      <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">
                        Recent Average Score
                      </div>
                      <div className="text-lg font-bold text-gray-900 dark:text-white">
                        {avgScore}
                      </div>
                    </div>
                  )}
                  
                  <button
                    onClick={() => handleStartTraining(module.id)}
                    disabled={activeModule === module.id}
                    className={`mt-4 w-full py-2 px-4 rounded-lg font-medium transition-colors ${
                      activeModule === module.id
                        ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
                        : 'bg-blue-600 text-white hover:bg-blue-700'
                    }`}
                  >
                    {activeModule === module.id ? 'Active' : 'Start Training'}
                  </button>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Achievements */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center space-x-3 mb-6">
          <Award className="h-6 w-6 text-yellow-500" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Achievements
          </h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {achievements.map((achievement, index) => (
            <motion.div
              key={achievement.title}
              className={`p-4 rounded-lg ${
                achievement.unlocked 
                  ? 'bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800' 
                  : 'bg-gray-50 dark:bg-gray-700'
              }`}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  {achievement.unlocked && (
                    <Trophy className="h-5 w-5 text-yellow-500" />
                  )}
                  <h4 className={`font-medium ${
                    achievement.unlocked 
                      ? 'text-yellow-800 dark:text-yellow-200' 
                      : 'text-gray-900 dark:text-white'
                  }`}>
                    {achievement.title}
                  </h4>
                </div>
                <span className="text-sm text-gray-500">
                  {Math.round(achievement.progress)}%
                </span>
              </div>
              <p className={`text-sm mb-2 ${
                achievement.unlocked 
                  ? 'text-yellow-700 dark:text-yellow-300' 
                  : 'text-gray-600 dark:text-gray-300'
              }`}>
                {achievement.description}
              </p>
              <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                <motion.div
                  className={`h-2 rounded-full ${
                    achievement.unlocked ? 'bg-yellow-500' : 'bg-gray-400'
                  }`}
                  initial={{ width: 0 }}
                  animate={{ width: `${achievement.progress}%` }}
                  transition={{ duration: 1, delay: index * 0.2 }}
                />
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Recent Sessions */}
      {sessions.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Recent Training Sessions
          </h3>
          <div className="space-y-3">
            {sessions.slice(-5).reverse().map((session) => {
              const module = trainingModules.find(m => m.id === session.moduleId);
              return (
                <div key={session.id} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <div className="flex items-center space-x-3">
                    {module && (
                      <div className={`p-2 rounded ${module.color}`}>
                        <module.icon className="h-4 w-4 text-white" />
                      </div>
                    )}
                    <div>
                      <div className="font-medium text-gray-900 dark:text-white">
                        {module?.title}
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {new Date(session.startTime).toLocaleDateString()}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-bold text-gray-900 dark:text-white">
                      {session.score}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      {Math.round(session.performance.accuracy)}% accuracy
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default Training;