import React from 'react';
import { motion } from 'framer-motion';
import { Lightbulb, Target, Coffee, Moon, Headphones, BookOpen, AlertCircle, TrendingUp } from 'lucide-react';
import { useEEGData } from '../hooks/useEEGData';

const Recommendations: React.FC = () => {
  const { mentalState, currentData } = useEEGData();

  const getRecommendations = () => {
    if (!currentData) return [];

    const recommendations = [];

    // Focus-based recommendations
    if (currentData.attention < 0.5) {
      recommendations.push({
        type: 'immediate',
        icon: Target,
        title: 'Improve Focus',
        description: 'Your attention level is below optimal. Try the Focus Enhancer training module.',
        action: 'Start Focus Training',
        priority: 'high',
        color: 'bg-red-500'
      });
    }

    // Stress-based recommendations
    if (currentData.beta > 0.7) {
      recommendations.push({
        type: 'wellness',
        icon: Moon,
        title: 'Reduce Stress',
        description: 'High beta activity detected. Consider taking a short meditation break.',
        action: 'Start Meditation',
        priority: 'medium',
        color: 'bg-orange-500'
      });
    }

    // Energy recommendations
    if (currentData.theta > 0.6) {
      recommendations.push({
        type: 'energy',
        icon: Coffee,
        title: 'Boost Energy',
        description: 'You seem drowsy. Try some light exercise or grab a healthy snack.',
        action: 'View Energy Tips',
        priority: 'medium',
        color: 'bg-yellow-500'
      });
    }

    // Positive recommendations
    if (currentData.attention > 0.7 && currentData.meditation > 0.6) {
      recommendations.push({
        type: 'opportunity',
        icon: TrendingUp,
        title: 'Peak Performance',
        description: 'You\'re in an optimal state! This is perfect for challenging tasks.',
        action: 'View Learning Materials',
        priority: 'low',
        color: 'bg-green-500'
      });
    }

    // Learning recommendations
    if (currentData.alpha > 0.6) {
      recommendations.push({
        type: 'learning',
        icon: BookOpen,
        title: 'Learning Opportunity',
        description: 'High alpha activity is great for creativity and learning new concepts.',
        action: 'Explore Brain Education',
        priority: 'low',
        color: 'bg-blue-500'
      });
    }

    // Audio recommendations
    if (currentData.quality < 0.8) {
      recommendations.push({
        type: 'technical',
        icon: Headphones,
        title: 'Signal Quality',
        description: 'EEG signal quality could be improved. Check your headset connection.',
        action: 'Check Device',
        priority: 'high',
        color: 'bg-purple-500'
      });
    }

    return recommendations;
  };

  const recommendations = getRecommendations();

  const articles = [
    {
      title: 'The Science of Flow States',
      description: 'How to achieve and maintain optimal performance states using EEG feedback.',
      readTime: '5 min read',
      category: 'Performance',
      image: 'https://images.pexels.com/photos/3825586/pexels-photo-3825586.jpeg?auto=compress&cs=tinysrgb&w=300'
    },
    {
      title: 'Meditation and Brainwaves',
      description: 'Understanding how different meditation practices affect your brainwave patterns.',
      readTime: '8 min read',
      category: 'Wellness',
      image: 'https://images.pexels.com/photos/3820296/pexels-photo-3820296.jpeg?auto=compress&cs=tinysrgb&w=300'
    },
    {
      title: 'Biofeedback for Better Sleep',
      description: 'Using neurofeedback techniques to improve sleep quality and duration.',
      readTime: '6 min read',
      category: 'Sleep',
      image: 'https://images.pexels.com/photos/3771115/pexels-photo-3771115.jpeg?auto=compress&cs=tinysrgb&w=300'
    }
  ];

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'border-red-200 dark:border-red-800';
      case 'medium': return 'border-orange-200 dark:border-orange-800';
      case 'low': return 'border-green-200 dark:border-green-800';
      default: return 'border-gray-200 dark:border-gray-700';
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Smart Recommendations
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Personalized suggestions based on your current brainwave patterns and mental state
        </p>
      </div>

      {/* Current State Overview */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
        <div className="flex items-center space-x-4">
          <div className={`p-3 rounded-full ${mentalState.color}`}>
            <Lightbulb className="h-6 w-6 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100">
              Current Mental State: {mentalState.primary}
            </h3>
            <p className="text-blue-700 dark:text-blue-300">
              {mentalState.description}
            </p>
            <div className="flex items-center space-x-2 mt-2">
              <div className="w-32 bg-blue-200 dark:bg-blue-800 rounded-full h-2">
                <motion.div
                  className="bg-blue-500 h-2 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${mentalState.confidence}%` }}
                  transition={{ duration: 1 }}
                />
              </div>
              <span className="text-sm font-medium text-blue-900 dark:text-blue-100">
                {mentalState.confidence}% confidence
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          Recommended Actions
        </h2>
        {recommendations.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {recommendations.map((rec, index) => {
              const Icon = rec.icon;
              return (
                <motion.div
                  key={index}
                  className={`bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border ${getPriorityColor(rec.priority)}`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  whileHover={{ scale: 1.02, y: -2 }}
                >
                  <div className="flex items-start space-x-4">
                    <div className={`p-3 rounded-lg ${rec.color}`}>
                      <Icon className="h-5 w-5 text-white" />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-2">
                        <h3 className="font-semibold text-gray-900 dark:text-white">
                          {rec.title}
                        </h3>
                        {rec.priority === 'high' && (
                          <AlertCircle className="h-4 w-4 text-red-500" />
                        )}
                      </div>
                      <p className="text-gray-600 dark:text-gray-300 text-sm mb-4">
                        {rec.description}
                      </p>
                      <button className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors">
                        {rec.action}
                      </button>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        ) : (
          <div className="bg-white dark:bg-gray-800 rounded-xl p-8 text-center shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="text-green-500 mb-4">
              <TrendingUp className="h-12 w-12 mx-auto" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              Great job! No immediate actions needed.
            </h3>
            <p className="text-gray-600 dark:text-gray-300">
              Your brainwave patterns look optimal. Keep up the good work!
            </p>
          </div>
        )}
      </div>

      {/* Recommended Reading */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          Recommended Reading
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {articles.map((article, index) => (
            <motion.div
              key={index}
              className="bg-white dark:bg-gray-800 rounded-xl overflow-hidden shadow-sm border border-gray-200 dark:border-gray-700"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.5 + index * 0.1 }}
              whileHover={{ scale: 1.02, y: -2 }}
            >
              <img
                src={article.image}
                alt={article.title}
                className="w-full h-40 object-cover"
              />
              <div className="p-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs bg-blue-100 dark:bg-blue-900/20 text-blue-800 dark:text-blue-400 px-2 py-1 rounded">
                    {article.category}
                  </span>
                  <span className="text-xs text-gray-500">{article.readTime}</span>
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                  {article.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-300 text-sm mb-4">
                  {article.description}
                </p>
                <button className="text-blue-600 dark:text-blue-400 text-sm font-medium hover:text-blue-700 dark:hover:text-blue-300">
                  Read More →
                </button>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Recommendations;