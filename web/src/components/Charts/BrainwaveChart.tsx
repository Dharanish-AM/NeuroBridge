import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { EEGData } from '../../types/eeg';

interface BrainwaveChartProps {
  data: EEGData[];
  height?: number;
}

const BrainwaveChart: React.FC<BrainwaveChartProps> = ({ data, height = 300 }) => {
  const chartData = data.map((d, index) => ({
    time: index,
    Alpha: d.alpha,
    Beta: d.beta,
    Gamma: d.gamma,
    Delta: d.delta,
    Theta: d.theta,
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={chartData.slice(-50)}>
        <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
        <XAxis 
          dataKey="time" 
          axisLine={false}
          tickLine={false}
          tick={{ fontSize: 12 }}
        />
        <YAxis 
          domain={[0, 1]}
          axisLine={false}
          tickLine={false}
          tick={{ fontSize: 12 }}
        />
        <Tooltip 
          contentStyle={{
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            border: 'none',
            borderRadius: '8px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
          }}
        />
        <Legend />
        <Line 
          type="monotone" 
          dataKey="Alpha" 
          stroke="#3B82F6" 
          strokeWidth={2}
          dot={false}
          strokeDasharray="none"
        />
        <Line 
          type="monotone" 
          dataKey="Beta" 
          stroke="#EF4444" 
          strokeWidth={2}
          dot={false}
        />
        <Line 
          type="monotone" 
          dataKey="Gamma" 
          stroke="#10B981" 
          strokeWidth={2}
          dot={false}
        />
        <Line 
          type="monotone" 
          dataKey="Delta" 
          stroke="#8B5CF6" 
          strokeWidth={2}
          dot={false}
        />
        <Line 
          type="monotone" 
          dataKey="Theta" 
          stroke="#F59E0B" 
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default BrainwaveChart;