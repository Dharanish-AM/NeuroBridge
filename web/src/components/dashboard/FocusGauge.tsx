import React, { useEffect, useState } from 'react';
import { Progress } from '@/components/ui/progress';

interface FocusGaugeProps {
  value: number;
}

const FocusGauge: React.FC<FocusGaugeProps> = ({ value }) => {
  const [animatedValue, setAnimatedValue] = useState(0);
  
  // Create a smooth animation for the gauge
  useEffect(() => {
    const timeout = setTimeout(() => {
      setAnimatedValue(value);
    }, 100);
    
    return () => clearTimeout(timeout);
  }, [value]);

  // Calculate the rotation for the gauge needle
  const needleRotation = (animatedValue / 100) * 180 - 90; // -90 to +90 degrees

  // Determine color based on focus level
  const getColor = () => {
    if (value >= 70) return 'bg-green-500';
    if (value >= 40) return 'bg-amber-500';
    return 'bg-red-500';
  };

  // Determine text color based on focus level
  const getTextColor = () => {
    if (value >= 70) return 'text-green-500';
    if (value >= 40) return 'text-amber-500';
    return 'text-red-500';
  };

  // Helper function to generate tick marks
  const generateTickMarks = () => {
    const ticks = [];
    for (let i = 0; i <= 180; i += 9) {  // Every 9 degrees (20 ticks)
      const isLargeTick = i % 45 === 0;  // Make every 5th tick larger
      const tickLength = isLargeTick ? '8px' : '4px';
      const tickThickness = isLargeTick ? '2px' : '1px';
      
      ticks.push(
        <div 
          key={i} 
          className="absolute bg-gray-300 origin-bottom" 
          style={{
            height: tickLength,
            width: tickThickness,
            transform: `rotate(${i - 90}deg)`,
            left: 'calc(50% - 1px)',
            bottom: '0'
          }}
        />
      );
    }
    return ticks;
  };

  // Generate labels for the gauge
  const generateLabels = () => {
    const labels = [0, 25, 50, 75, 100];
    return labels.map(label => {
      // Calculate position based on percentage
      const degrees = (label / 100) * 180 - 90;
      const radians = degrees * (Math.PI / 180);
      
      // Position around the gauge arc
      const radius = 132;
      const x = Math.cos(radians) * radius;
      const y = Math.sin(radians) * radius;
      
      return (
        <div 
          key={label}
          className="absolute text-xs text-gray-500"
          style={{
            transform: `translate(-50%, -50%)`,
            left: `calc(50% + ${x}px)`,
            bottom: `calc(50% + ${y}px)`
          }}
        >
          {label}%
        </div>
      );
    });
  };

  return (
    <div className="w-64 h-64 relative flex flex-col items-center">
      {/* Gauge labels */}
      {generateLabels()}
      
      {/* Semi-circular gauge */}
      <div className="w-full h-32 bg-gray-100 rounded-t-full overflow-hidden relative">
        {/* Tick marks */}
        {generateTickMarks()}
        
        {/* Colored fill */}
        <div 
          className={`h-full ${getColor()} transition-all duration-700 ease-out`}
          style={{ 
            width: `${animatedValue}%`,
            borderRadius: '100% 100% 0 0'
          }}
        ></div>
      </div>
      
      {/* Gauge needle */}
      <div 
        className="absolute top-32 left-32 w-1 h-32 bg-gray-800 origin-bottom transform transition-transform duration-700 ease-out"
        style={{ transform: `translateX(-50%) rotate(${needleRotation}deg)` }}
      >
        {/* Needle tip */}
        <div className="w-2 h-2 bg-gray-800 absolute -top-1 -left-0.5 rounded-full"></div>
      </div>
      
      {/* Center point of needle */}
      <div className="absolute top-32 left-32 w-6 h-6 bg-white border-2 border-gray-800 rounded-full transform -translate-x-1/2 -translate-y-1/2 z-10"></div>
      
      {/* Value display */}
      <div className={`mt-8 text-3xl font-bold ${getTextColor()}`}>{animatedValue}%</div>
      <div className="text-sm text-gray-500">Focus Level</div>
      
      {/* Status text */}
      <div className="mt-2 text-sm font-medium bg-gray-100 px-3 py-1 rounded-full">
        {value >= 70 ? 'High Focus - Optimal learning state' : 
         value >= 40 ? 'Medium Focus - Good for learning' : 
         'Low Focus - Consider taking a break'}
      </div>
    </div>
  );
};

export default FocusGauge;