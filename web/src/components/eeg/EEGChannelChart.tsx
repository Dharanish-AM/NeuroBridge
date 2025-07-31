import React from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  ResponsiveContainer, 
  CartesianGrid, 
  ReferenceLine 
} from 'recharts';
import { ChannelData } from '@/context/BrainDataContext';

interface EEGChannelChartProps {
  channel: ChannelData;
  filtered: boolean;
}

const EEGChannelChart: React.FC<EEGChannelChartProps> = ({ channel, filtered }) => {
  // Process raw data into chart-friendly format
  const chartData = channel.history.map((value, index) => ({
    time: index,
    raw: value,
    filtered: applyFilter(value, index, channel.history) // Simple filtering function
  }));
  
  // Choose channel color based on name
  const getChannelColor = (name: string) => {
    if (name.startsWith('Fp')) return '#6366f1'; // Indigo for pre-frontal
    if (name.startsWith('F')) return '#8b5cf6'; // Purple for frontal
    if (name.startsWith('C')) return '#ec4899'; // Pink for central
    if (name.startsWith('T')) return '#14b8a6'; // Teal for temporal
    if (name.startsWith('O')) return '#f97316'; // Orange for occipital
    return '#94a3b8'; // Slate for others
  };
  
  return (
    <div className="h-full">
      <div className="flex justify-between items-center mb-1">
        <span className="font-medium" style={{ color: getChannelColor(channel.name) }}>
          {channel.name}
        </span>
        <span className="text-sm text-muted-foreground">
          {channel.value.toFixed(2)} µV
        </span>
      </div>
      <ResponsiveContainer width="100%" height="90%">
        <LineChart
          data={chartData.slice(-50)} // Show only the last 50 points
          margin={{ top: 5, right: 5, left: 0, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
          <XAxis dataKey="time" hide={true} />
          <YAxis domain={[-100, 100]} hide={true} />
          <ReferenceLine y={0} stroke="#cbd5e1" strokeWidth={1} />
          <Line
            type="monotone"
            dataKey={filtered ? "filtered" : "raw"}
            stroke={getChannelColor(channel.name)}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

// Simple low-pass filter function (moving average)
const applyFilter = (currentValue: number, index: number, history: number[]): number => {
  const windowSize = 5;
  let sum = currentValue;
  let count = 1;
  
  // Look back up to windowSize samples
  for (let i = 1; i <= windowSize; i++) {
    const idx = index - i;
    if (idx >= 0) {
      sum += history[idx];
      count++;
    }
  }
  
  return sum / count;
};

export default EEGChannelChart;