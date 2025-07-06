import React from 'react';
import { RadarChart as RechartsRadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts';
import { EEGData } from '../../types/eeg';

interface RadarChartProps {
  data: EEGData | null;
  height?: number;
}

const RadarChart: React.FC<RadarChartProps> = ({ data, height = 300 }) => {
  if (!data) return null;

  const radarData = [
    { wave: 'Alpha', value: data.alpha, fullMark: 1 },
    { wave: 'Beta', value: data.beta, fullMark: 1 },
    { wave: 'Gamma', value: data.gamma, fullMark: 1 },
    { wave: 'Delta', value: data.delta, fullMark: 1 },
    { wave: 'Theta', value: data.theta, fullMark: 1 },
  ];

  return (
    <ResponsiveContainer width="100%" height={height}>
      <RechartsRadarChart data={radarData}>
        <PolarGrid />
        <PolarAngleAxis dataKey="wave" />
        <PolarRadiusAxis 
          angle={90} 
          domain={[0, 1]} 
          tick={false}
          axisLine={false}
        />
        <Radar
          name="Brainwaves"
          dataKey="value"
          stroke="#3B82F6"
          fill="#3B82F6"
          fillOpacity={0.2}
          strokeWidth={2}
        />
      </RechartsRadarChart>
    </ResponsiveContainer>
  );
};

export default RadarChart;