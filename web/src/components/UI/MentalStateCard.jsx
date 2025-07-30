import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Zap } from 'lucide-react';

const MentalStateCard = ({ mentalState }) => {
  return (
    <motion.div
      className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="flex items-center space-x-4">
        <div className={`p-3 rounded-full ${mentalState.color}`}>
          <Brain className="h-8 w-8 text-white" />
        </div>
        <div className="flex-1">
          <div className="flex items-center space-x-2">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white">
              {mentalState.primary}
            </h3>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              • {mentalState.secondary}
            </span>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
            {mentalState.description}
          </p>
          <div className="flex items-center space-x-2 mt-3">
            <Zap className="h-4 w-4 text-yellow-500" />
            <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <motion.div
                className={`h-2 rounded-full ${mentalState.color}`}
                initial={{ width: 0 }}
                animate={{ width: `${mentalState.confidence}%` }}
                transition={{ duration: 1, ease: "easeOut" }}
              />
            </div>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              {mentalState.confidence}%
            </span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default MentalStateCard;