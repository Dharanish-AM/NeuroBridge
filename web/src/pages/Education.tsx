import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BookOpen, Play, CheckCircle, Brain, Zap, Waves, X } from 'lucide-react';
import LessonContent from '../components/Education/LessonContent';

const Education: React.FC = () => {
  const [activeLesson, setActiveLesson] = useState<string | null>(null);
  const [completedLessons, setCompletedLessons] = useState<string[]>(['alpha-waves']);

  const lessons = [
    {
      id: 'alpha-waves',
      title: 'Understanding Alpha Waves',
      description: 'Learn about alpha brainwaves and their role in relaxation and creativity',
      duration: '8 min',
      difficulty: 'Beginner',
      category: 'Brainwave Basics',
      color: 'bg-blue-500',
      content: {
        overview: 'Alpha waves (8-12 Hz) are associated with relaxed, calm states of mind. They represent a bridge between conscious thinking and the subconscious mind, often appearing when you\'re awake but relaxed, such as during light meditation or when your eyes are closed.',
        keyPoints: [
          'Generated when you\'re awake but relaxed, typically with eyes closed',
          'Associated with creativity, insight, and the "flow" state',
          'Appear during meditation and light sleep transitions',
          'Can be increased through practice and specific techniques',
          'Optimal for learning and memory consolidation'
        ],
        tips: [
          'Close your eyes and breathe deeply for 2-3 minutes',
          'Practice mindfulness meditation daily',
          'Listen to calming music or nature sounds',
          'Spend time in nature without distractions',
          'Try visualization exercises before sleep'
        ]
      }
    },
    {
      id: 'beta-waves',
      title: 'Beta Waves and Focus',
      description: 'Discover how beta waves affect concentration and alertness',
      duration: '10 min',
      difficulty: 'Beginner',
      category: 'Brainwave Basics',
      color: 'bg-red-500',
      content: {
        overview: 'Beta waves (12-30 Hz) are present during active, analytical thinking and normal waking consciousness. They are essential for concentration, alertness, and cognitive tasks, but excessive beta activity can lead to anxiety and stress.',
        keyPoints: [
          'Associated with focused attention and analytical thinking',
          'High during problem-solving and decision-making',
          'Can indicate stress and anxiety when excessive',
          'Normal and necessary during daily activities',
          'Divided into low, mid, and high beta frequencies'
        ],
        tips: [
          'Take regular breaks to prevent beta overactivity',
          'Practice deep breathing when feeling stressed',
          'Balance focused work with relaxation periods',
          'Maintain good sleep hygiene to regulate beta waves',
          'Use time-blocking for intense cognitive tasks'
        ]
      }
    },
    {
      id: 'theta-waves',
      title: 'Theta States and Creativity',
      description: 'Explore theta waves and their connection to creativity and learning',
      duration: '12 min',
      difficulty: 'Intermediate',
      category: 'Advanced Concepts',
      color: 'bg-yellow-500',
      content: {
        overview: 'Theta waves (4-8 Hz) occur during deep meditation, REM sleep, and states of deep creativity. They are associated with access to the subconscious mind, enhanced learning, and profound insights.',
        keyPoints: [
          'Associated with creativity, intuition, and deep insights',
          'Present during deep meditation and REM sleep',
          'Important for memory consolidation and learning',
          'Can enhance problem-solving abilities',
          'Linked to emotional processing and healing'
        ],
        tips: [
          'Practice deep meditation regularly (20+ minutes)',
          'Try visualization and guided imagery exercises',
          'Engage in creative activities like art or music',
          'Maintain consistent sleep patterns for natural theta',
          'Use binaural beats in the theta range (4-8 Hz)'
        ]
      }
    },
    {
      id: 'gamma-waves',
      title: 'Gamma Waves and Consciousness',
      description: 'Advanced topic on gamma waves and higher consciousness states',
      duration: '15 min',
      difficulty: 'Advanced',
      category: 'Advanced Concepts',
      color: 'bg-green-500',
      content: {
        overview: 'Gamma waves (30-100 Hz) are the fastest brainwaves and are linked to high-level cognitive functioning, consciousness, and peak mental performance. They represent the binding of different brain regions into a unified conscious experience.',
        keyPoints: [
          'Associated with conscious awareness and peak performance',
          'Present during moments of insight and "aha!" experiences',
          'Linked to feelings of compassion and universal love',
          'Increased in experienced meditators and monks',
          'May play a role in memory formation and recall'
        ],
        tips: [
          'Practice loving-kindness meditation regularly',
          'Engage in complex problem-solving activities',
          'Maintain a consistent meditation practice',
          'Challenge yourself with new learning experiences',
          'Cultivate mindfulness in daily activities'
        ]
      }
    },
    {
      id: 'delta-waves',
      title: 'Delta Waves and Deep Sleep',
      description: 'Understanding delta waves and their role in restorative sleep',
      duration: '9 min',
      difficulty: 'Beginner',
      category: 'Sleep & Recovery',
      color: 'bg-purple-500',
      content: {
        overview: 'Delta waves (0.5-4 Hz) are the slowest brainwaves and are dominant during deep, dreamless sleep. They are crucial for physical restoration, immune function, and memory consolidation.',
        keyPoints: [
          'Dominant during deep, restorative sleep stages',
          'Essential for physical healing and recovery',
          'Important for immune system function',
          'Facilitate memory consolidation',
          'Decrease with age, affecting sleep quality'
        ],
        tips: [
          'Maintain a consistent sleep schedule',
          'Create a cool, dark sleeping environment',
          'Avoid screens 1-2 hours before bedtime',
          'Practice relaxation techniques before sleep',
          'Consider meditation or yoga nidra practices'
        ]
      }
    },
    {
      id: 'neurofeedback-basics',
      title: 'Introduction to Neurofeedback',
      description: 'Learn how neurofeedback training can optimize brain function',
      duration: '14 min',
      difficulty: 'Intermediate',
      category: 'Training Methods',
      color: 'bg-indigo-500',
      content: {
        overview: 'Neurofeedback is a form of biofeedback that uses real-time displays of brain activity to teach self-regulation of brain function. It\'s a non-invasive method that can help improve focus, reduce anxiety, and enhance overall mental performance.',
        keyPoints: [
          'Uses real-time EEG feedback to train brain patterns',
          'Non-invasive and drug-free approach to brain training',
          'Can help with ADHD, anxiety, and sleep disorders',
          'Requires consistent practice for lasting results',
          'Based on principles of operant conditioning'
        ],
        tips: [
          'Start with short 10-15 minute sessions',
          'Be consistent with training schedule',
          'Stay relaxed and avoid forcing changes',
          'Track your progress over time',
          'Combine with other wellness practices'
        ]
      }
    }
  ];

  const categories = Array.from(new Set(lessons.map(lesson => lesson.category)));

  const handleStartLesson = (lessonId: string) => {
    setActiveLesson(lessonId);
  };

  const handleCompleteLesson = () => {
    if (activeLesson && !completedLessons.includes(activeLesson)) {
      setCompletedLessons([...completedLessons, activeLesson]);
    }
    setActiveLesson(null);
  };

  const activeLessonData = lessons.find(lesson => lesson.id === activeLesson);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Brain Education Hub
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Learn about brainwaves, neuroscience, and how to optimize your mental performance
        </p>
      </div>

      {/* Progress Overview */}
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 border border-purple-200 dark:border-purple-800">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-purple-900 dark:text-purple-100">
              Learning Progress
            </h3>
            <p className="text-purple-700 dark:text-purple-300">
              {completedLessons.length} of {lessons.length} lessons completed
            </p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-purple-900 dark:text-purple-100">
              {Math.round((completedLessons.length / lessons.length) * 100)}%
            </div>
            <div className="w-32 bg-purple-200 dark:bg-purple-800 rounded-full h-2 mt-2">
              <motion.div
                className="bg-purple-500 h-2 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${(completedLessons.length / lessons.length) * 100}%` }}
                transition={{ duration: 1 }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Active Lesson Modal */}
      <AnimatePresence>
        {activeLesson && activeLessonData && (
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
                  <div className={`p-2 rounded-lg ${activeLessonData.color}`}>
                    <Brain className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                      {activeLessonData.title}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400">
                      {activeLessonData.category} • {activeLessonData.duration}
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setActiveLesson(null)}
                  className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                >
                  <X className="h-6 w-6" />
                </button>
              </div>

              <LessonContent
                lesson={activeLessonData}
                onComplete={handleCompleteLesson}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Lesson Categories */}
      {categories.map((category, categoryIndex) => (
        <div key={category} className="space-y-4">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            {category}
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {lessons
              .filter(lesson => lesson.category === category)
              .map((lesson, lessonIndex) => {
                const isCompleted = completedLessons.includes(lesson.id);
                return (
                  <motion.div
                    key={lesson.id}
                    className={`bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border transition-colors ${
                      isCompleted
                        ? 'border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/10'
                        : 'border-gray-200 dark:border-gray-700'
                    }`}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: (categoryIndex * 0.1) + (lessonIndex * 0.05) }}
                    whileHover={{ scale: 1.02, y: -2 }}
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div className={`p-2 rounded-lg ${lesson.color}`}>
                        <BookOpen className="h-5 w-5 text-white" />
                      </div>
                      {isCompleted && (
                        <CheckCircle className="h-5 w-5 text-green-500" />
                      )}
                    </div>
                    
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                      {lesson.title}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-300 text-sm mb-4">
                      {lesson.description}
                    </p>
                    
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center space-x-4">
                        <span className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                          {lesson.difficulty}
                        </span>
                        <span className="text-xs text-gray-500">{lesson.duration}</span>
                      </div>
                    </div>
                    
                    <button
                      onClick={() => handleStartLesson(lesson.id)}
                      disabled={activeLesson === lesson.id}
                      className={`w-full py-2 px-4 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2 ${
                        isCompleted
                          ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
                          : activeLesson === lesson.id
                          ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
                          : 'bg-blue-600 text-white hover:bg-blue-700'
                      }`}
                    >
                      {isCompleted ? (
                        <>
                          <CheckCircle className="h-4 w-4" />
                          <span>Review Lesson</span>
                        </>
                      ) : activeLesson === lesson.id ? (
                        <span>Active</span>
                      ) : (
                        <>
                          <Play className="h-4 w-4" />
                          <span>Start Lesson</span>
                        </>
                      )}
                    </button>
                  </motion.div>
                );
              })}
          </div>
        </div>
      ))}
    </div>
  );
};

export default Education;