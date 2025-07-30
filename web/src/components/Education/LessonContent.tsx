import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight, Play, Pause, RotateCcw, Brain, Zap, Waves } from 'lucide-react';
import { useEEGData } from '../../hooks/useEEGData';

interface LessonContentProps {
  lesson: {
    id: string;
    title: string;
    content: {
      overview: string;
      keyPoints: string[];
      tips: string[];
      interactive?: {
        type: 'wave-simulator' | 'breathing-exercise' | 'focus-test';
        config: any;
      };
    };
  };
  onComplete: () => void;
}

const LessonContent: React.FC<LessonContentProps> = ({ lesson, onComplete }) => {
  const [currentSection, setCurrentSection] = useState(0);
  const [progress, setProgress] = useState(0);
  const [isInteractiveActive, setIsInteractiveActive] = useState(false);
  const [simulatedWaves, setSimulatedWaves] = useState({
    alpha: 0.5,
    beta: 0.3,
    gamma: 0.2,
    delta: 0.7,
    theta: 0.4
  });
  const { currentData } = useEEGData();

  const sections = [
    { title: 'Overview', type: 'text' },
    { title: 'Key Points', type: 'points' },
    { title: 'Interactive Demo', type: 'interactive' },
    { title: 'Practical Tips', type: 'tips' },
    { title: 'Summary', type: 'summary' }
  ];

  useEffect(() => {
    setProgress((currentSection / (sections.length - 1)) * 100);
  }, [currentSection, sections.length]);

  const WaveSimulator = () => {
    const [selectedWave, setSelectedWave] = useState<keyof typeof simulatedWaves>('alpha');
    
    const waveDescriptions = {
      alpha: 'Alpha waves (8-12 Hz) - Associated with relaxed awareness and creativity',
      beta: 'Beta waves (12-30 Hz) - Present during active thinking and concentration',
      gamma: 'Gamma waves (30-100 Hz) - Linked to high-level cognitive processing',
      delta: 'Delta waves (0.5-4 Hz) - Dominant during deep sleep',
      theta: 'Theta waves (4-8 Hz) - Present during meditation and REM sleep'
    };

    return (
      <div className="space-y-6">
        <div className="text-center">
          <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            Brainwave Simulator
          </h4>
          <p className="text-gray-600 dark:text-gray-300 text-sm">
            Adjust the sliders to see how different brainwave patterns affect mental states
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            {Object.entries(simulatedWaves).map(([wave, value]) => (
              <div key={wave} className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium text-gray-900 dark:text-white capitalize">
                    {wave}
                  </label>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {Math.round(value * 100)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={value}
                  onChange={(e) => setSimulatedWaves(prev => ({
                    ...prev,
                    [wave]: parseFloat(e.target.value)
                  }))}
                  className="w-full"
                />
              </div>
            ))}
          </div>
          
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <h5 className="font-medium text-gray-900 dark:text-white mb-3">
              Current Mental State
            </h5>
            <div className="space-y-2">
              {simulatedWaves.alpha > 0.6 && simulatedWaves.theta > 0.5 && (
                <div className="flex items-center space-x-2 text-green-600">
                  <Brain className="h-4 w-4" />
                  <span className="text-sm">Deep Meditation</span>
                </div>
              )}
              {simulatedWaves.beta > 0.7 && (
                <div className="flex items-center space-x-2 text-blue-600">
                  <Zap className="h-4 w-4" />
                  <span className="text-sm">High Focus</span>
                </div>
              )}
              {simulatedWaves.delta > 0.8 && (
                <div className="flex items-center space-x-2 text-purple-600">
                  <Waves className="h-4 w-4" />
                  <span className="text-sm">Deep Sleep State</span>
                </div>
              )}
              {simulatedWaves.gamma > 0.6 && simulatedWaves.beta > 0.5 && (
                <div className="flex items-center space-x-2 text-orange-600">
                  <Brain className="h-4 w-4" />
                  <span className="text-sm">Peak Performance</span>
                </div>
              )}
            </div>
            
            <div className="mt-4 p-3 bg-white dark:bg-gray-800 rounded border">
              <p className="text-xs text-gray-600 dark:text-gray-400">
                {waveDescriptions[selectedWave]}
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const BreathingExercise = () => {
    const [isActive, setIsActive] = useState(false);
    const [phase, setPhase] = useState<'inhale' | 'hold' | 'exhale'>('inhale');
    const [phaseTime, setPhaseTime] = useState(4);

    useEffect(() => {
      if (isActive) {
        const timer = setInterval(() => {
          setPhaseTime(t => {
            if (t <= 1) {
              if (phase === 'inhale') {
                setPhase('hold');
                return 2;
              } else if (phase === 'hold') {
                setPhase('exhale');
                return 6;
              } else {
                setPhase('inhale');
                return 4;
              }
            }
            return t - 1;
          });
        }, 1000);

        return () => clearInterval(timer);
      }
    }, [isActive, phase]);

    return (
      <div className="text-center space-y-6">
        <h4 className="text-lg font-semibold text-gray-900 dark:text-white">
          Breathing Exercise
        </h4>
        <p className="text-gray-600 dark:text-gray-300 text-sm">
          Follow the breathing pattern to see how it affects your brainwaves
        </p>
        
        <div className="relative">
          <motion.div
            className={`w-32 h-32 mx-auto rounded-full bg-gradient-to-br ${
              phase === 'inhale' ? 'from-blue-400 to-blue-600' :
              phase === 'hold' ? 'from-purple-400 to-purple-600' :
              'from-green-400 to-green-600'
            } flex items-center justify-center text-white`}
            animate={{
              scale: phase === 'inhale' ? 1.2 : phase === 'hold' ? 1.2 : 0.8,
            }}
            transition={{ duration: 1, ease: "easeInOut" }}
          >
            <div className="text-center">
              <div className="text-lg font-bold">
                {phase === 'inhale' ? 'Breathe In' : 
                 phase === 'hold' ? 'Hold' : 'Breathe Out'}
              </div>
              <div className="text-sm">{phaseTime}</div>
            </div>
          </motion.div>
        </div>

        <button
          onClick={() => setIsActive(!isActive)}
          className={`px-6 py-2 rounded-lg font-medium transition-colors ${
            isActive 
              ? 'bg-red-500 hover:bg-red-600 text-white' 
              : 'bg-blue-500 hover:bg-blue-600 text-white'
          }`}
        >
          {isActive ? 'Stop' : 'Start'} Exercise
        </button>

        {isActive && (
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <p className="text-sm text-blue-800 dark:text-blue-200">
              Notice how controlled breathing can help regulate your mental state and 
              promote alpha wave activity associated with relaxation.
            </p>
          </div>
        )}
      </div>
    );
  };

  const renderSection = () => {
    const section = sections[currentSection];
    
    switch (section.type) {
      case 'text':
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
              {lesson.title}
            </h3>
            <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
              {lesson.content.overview}
            </p>
          </motion.div>
        );
        
      case 'points':
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
              Key Learning Points
            </h3>
            <ul className="space-y-3">
              {lesson.content.keyPoints.map((point, index) => (
                <motion.li
                  key={index}
                  className="flex items-start space-x-3"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Zap className="h-5 w-5 text-blue-500 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-600 dark:text-gray-300">{point}</span>
                </motion.li>
              ))}
            </ul>
          </motion.div>
        );
        
      case 'interactive':
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
              Interactive Learning
            </h3>
            {lesson.id === 'alpha-waves' && <BreathingExercise />}
            {lesson.id !== 'alpha-waves' && <WaveSimulator />}
          </motion.div>
        );
        
      case 'tips':
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
              Practical Tips
            </h3>
            <ul className="space-y-3">
              {lesson.content.tips.map((tip, index) => (
                <motion.li
                  key={index}
                  className="flex items-start space-x-3"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Waves className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-600 dark:text-gray-300">{tip}</span>
                </motion.li>
              ))}
            </ul>
          </motion.div>
        );
        
      case 'summary':
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6 text-center"
          >
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
              Lesson Complete!
            </h3>
            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
              <div className="text-green-800 dark:text-green-200 space-y-2">
                <p className="font-medium">You've successfully learned about {lesson.title}</p>
                <p className="text-sm">
                  You can now apply this knowledge to better understand your brainwave patterns
                  and optimize your mental performance.
                </p>
              </div>
            </div>
            <button
              onClick={onComplete}
              className="bg-blue-600 text-white px-8 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors"
            >
              Complete Lesson
            </button>
          </motion.div>
        );
        
      default:
        return null;
    }
  };

  return (
    <div className="space-y-6">
      {/* Progress Bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {sections[currentSection].title}
          </span>
          <span className="text-sm text-gray-600 dark:text-gray-400">
            {currentSection + 1} of {sections.length}
          </span>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <motion.div
            className="bg-blue-500 h-2 rounded-full"
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </div>

      {/* Content */}
      <div className="min-h-96">
        <AnimatePresence mode="wait">
          {renderSection()}
        </AnimatePresence>
      </div>

      {/* Navigation */}
      <div className="flex items-center justify-between pt-6 border-t border-gray-200 dark:border-gray-700">
        <button
          onClick={() => setCurrentSection(Math.max(0, currentSection - 1))}
          disabled={currentSection === 0}
          className="flex items-center space-x-2 px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <ChevronLeft className="h-4 w-4" />
          <span>Previous</span>
        </button>

        <div className="flex space-x-2">
          {sections.map((_, index) => (
            <button
              key={index}
              onClick={() => setCurrentSection(index)}
              className={`w-3 h-3 rounded-full transition-colors ${
                index === currentSection 
                  ? 'bg-blue-500' 
                  : index < currentSection 
                  ? 'bg-green-500' 
                  : 'bg-gray-300 dark:bg-gray-600'
              }`}
            />
          ))}
        </div>

        <button
          onClick={() => setCurrentSection(Math.min(sections.length - 1, currentSection + 1))}
          disabled={currentSection === sections.length - 1}
          className="flex items-center space-x-2 px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <span>Next</span>
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
};

export default LessonContent;