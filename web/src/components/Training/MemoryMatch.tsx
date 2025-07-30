import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Star, Zap, Heart, Diamond, Circle } from 'lucide-react';
import { useEEGData } from '../../hooks/useEEGData';

interface MemoryMatchProps {
  onComplete: (score: number, performance: any) => void;
  onUpdateScore: (points: number) => void;
}

const MemoryMatch: React.FC<MemoryMatchProps> = ({ onComplete, onUpdateScore }) => {
  const [cards, setCards] = useState<Array<{id: number, icon: any, matched: boolean, flipped: boolean}>>([]);
  const [flippedCards, setFlippedCards] = useState<number[]>([]);
  const [matches, setMatches] = useState(0);
  const [attempts, setAttempts] = useState(0);
  const [timeLeft, setTimeLeft] = useState(120); // 2 minutes
  const [level, setLevel] = useState(1);
  const [score, setScore] = useState(0);
  const [reactionTimes, setReactionTimes] = useState<number[]>([]);
  const [cardFlipTime, setCardFlipTime] = useState(0);
  const { currentData } = useEEGData();

  const icons = [Brain, Star, Zap, Heart, Diamond, Circle];

  const initializeCards = useCallback((level: number) => {
    const pairCount = Math.min(3 + level, 8); // 3-8 pairs based on level
    const selectedIcons = icons.slice(0, pairCount);
    const cardPairs = [...selectedIcons, ...selectedIcons];
    
    const shuffledCards = cardPairs
      .map((icon, index) => ({
        id: index,
        icon,
        matched: false,
        flipped: false
      }))
      .sort(() => Math.random() - 0.5);
    
    setCards(shuffledCards);
    setFlippedCards([]);
    setMatches(0);
    setAttempts(0);
  }, []);

  useEffect(() => {
    initializeCards(level);
  }, [level, initializeCards]);

  useEffect(() => {
    if (timeLeft > 0) {
      const timer = setTimeout(() => setTimeLeft(t => t - 1), 1000);
      return () => clearTimeout(timer);
    } else {
      // Game over
      const accuracy = attempts > 0 ? (matches / attempts) * 100 : 0;
      const avgReactionTime = reactionTimes.length > 0 
        ? reactionTimes.reduce((a, b) => a + b, 0) / reactionTimes.length 
        : 0;
      const focusLevel = (currentData?.attention || 0) * 100;
      
      onComplete(score, {
        accuracy,
        reactionTime: avgReactionTime,
        consistency: Math.max(0, 100 - (level - 1) * 10), // Consistency based on level reached
        focusLevel
      });
    }
  }, [timeLeft, score, matches, attempts, reactionTimes, currentData, level, onComplete]);

  const handleCardClick = useCallback((cardId: number) => {
    if (flippedCards.length >= 2 || flippedCards.includes(cardId) || cards[cardId]?.matched) {
      return;
    }

    const now = Date.now();
    if (flippedCards.length === 0) {
      setCardFlipTime(now);
    } else {
      const reactionTime = now - cardFlipTime;
      setReactionTimes(prev => [...prev, reactionTime]);
    }

    const newFlippedCards = [...flippedCards, cardId];
    setFlippedCards(newFlippedCards);

    setCards(prev => prev.map(card => 
      card.id === cardId ? { ...card, flipped: true } : card
    ));

    if (newFlippedCards.length === 2) {
      setAttempts(a => a + 1);
      
      const [firstId, secondId] = newFlippedCards;
      const firstCard = cards[firstId];
      const secondCard = cards[secondId];

      if (firstCard.icon === secondCard.icon) {
        // Match found!
        setMatches(m => m + 1);
        
        // Score based on speed and focus level
        const focusBonus = (currentData?.attention || 0.5) * 50;
        const speedBonus = Math.max(0, 30 - (reactionTimes[reactionTimes.length - 1] || 1000) / 100);
        const points = Math.round(20 + focusBonus + speedBonus);
        
        setScore(s => s + points);
        onUpdateScore(points);

        setTimeout(() => {
          setCards(prev => prev.map(card => 
            card.id === firstId || card.id === secondId 
              ? { ...card, matched: true } 
              : card
          ));
          setFlippedCards([]);
          
          // Check if level complete
          const newMatches = matches + 1;
          const totalPairs = cards.length / 2;
          if (newMatches === totalPairs) {
            // Level complete, advance to next level
            setTimeout(() => {
              setLevel(l => l + 1);
              setTimeLeft(t => t + 30); // Bonus time
            }, 1000);
          }
        }, 1000);
      } else {
        // No match
        setTimeout(() => {
          setCards(prev => prev.map(card => 
            card.id === firstId || card.id === secondId 
              ? { ...card, flipped: false } 
              : card
          ));
          setFlippedCards([]);
        }, 1500);
      }
    }
  }, [flippedCards, cards, matches, currentData, cardFlipTime, reactionTimes, onUpdateScore]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const accuracy = attempts > 0 ? Math.round((matches / attempts) * 100) : 0;

  return (
    <div className="relative w-full h-96 bg-gradient-to-br from-purple-50 to-pink-100 dark:from-purple-900/20 dark:to-pink-900/30 rounded-xl border border-purple-200 dark:border-purple-800 overflow-hidden">
      {/* Header */}
      <div className="absolute top-4 left-4 right-4 flex items-center justify-between z-10">
        <div className="flex items-center space-x-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg px-3 py-1 shadow-sm">
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              Level {level}
            </span>
          </div>
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
            className="bg-purple-500 h-2 rounded-full"
            animate={{ width: `${(currentData?.attention || 0) * 100}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </div>

      {/* Game Grid */}
      <div className="absolute inset-0 pt-20 pb-16 px-4">
        <div className={`grid gap-2 h-full ${
          cards.length <= 8 ? 'grid-cols-4' : 
          cards.length <= 12 ? 'grid-cols-4' : 'grid-cols-6'
        }`}>
          <AnimatePresence>
            {cards.map((card) => {
              const IconComponent = card.icon;
              return (
                <motion.button
                  key={card.id}
                  className={`relative rounded-lg shadow-md transition-all duration-300 ${
                    card.matched 
                      ? 'bg-green-200 dark:bg-green-800 cursor-default' 
                      : card.flipped 
                      ? 'bg-white dark:bg-gray-700' 
                      : 'bg-purple-200 dark:bg-purple-800 hover:bg-purple-300 dark:hover:bg-purple-700'
                  }`}
                  onClick={() => handleCardClick(card.id)}
                  disabled={card.matched || flippedCards.length >= 2}
                  whileHover={{ scale: card.matched ? 1 : 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  initial={{ rotateY: 0 }}
                  animate={{ rotateY: card.flipped || card.matched ? 180 : 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="absolute inset-0 flex items-center justify-center backface-hidden">
                    {card.flipped || card.matched ? (
                      <IconComponent className={`h-6 w-6 ${
                        card.matched ? 'text-green-600' : 'text-purple-600'
                      }`} />
                    ) : (
                      <div className="w-4 h-4 bg-purple-400 rounded-full"></div>
                    )}
                  </div>
                </motion.button>
              );
            })}
          </AnimatePresence>
        </div>
      </div>

      {/* Progress */}
      <div className="absolute bottom-4 left-4 right-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Progress: {matches}/{cards.length / 2} pairs
            </span>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              {(currentData?.attention || 0) > 0.7 
                ? "Great focus! Keep it up!" 
                : "Stay concentrated on the patterns."}
            </span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <motion.div
              className="bg-purple-500 h-2 rounded-full"
              animate={{ width: `${(matches / (cards.length / 2)) * 100}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default MemoryMatch;