import React, { useState } from 'react';
import { useBrainData } from '@/context/BrainDataContext';
import { 
  Bar, 
  BarChart, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  PieChart,
  Pie,
  Cell,
  Legend
} from 'recharts';
import { Card, CardContent } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

// Define types for chart labels
interface ChartLabel {
  cx: number;
  cy: number;
  midAngle: number;
  innerRadius: number;
  outerRadius: number;
  percent: number;
  index: number;
  name: string;
}

const BrainwavesChart: React.FC = () => {
  const { brainwaves } = useBrainData();
  const [chartType, setChartType] = useState<'bar' | 'pie'>('bar');
  
  const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }: ChartLabel) => {
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * Math.PI / 180);
    const y = cy + radius * Math.sin(-midAngle * Math.PI / 180);
  
    return (
      <text 
        x={x} 
        y={y} 
        fill="white" 
        textAnchor={x > cx ? 'start' : 'end'} 
        dominantBaseline="central"
        fontSize={12}
        fontWeight="bold"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  return (
    <div className="h-full">
      <div className="flex justify-end mb-4">
        <Tabs 
          defaultValue="bar" 
          value={chartType} 
          onValueChange={(value) => setChartType(value as 'bar' | 'pie')}
          className="w-[200px]"
        >
          <TabsList className="grid grid-cols-2">
            <TabsTrigger value="bar">Bar</TabsTrigger>
            <TabsTrigger value="pie">Pie</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      <div className="h-[250px]">
        {chartType === 'bar' ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={brainwaves} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <XAxis dataKey="name" />
              <YAxis domain={[0, 100]} />
              <Tooltip 
                formatter={(value) => [`${value}%`, 'Power']}
                cursor={{ fill: 'rgba(0, 0, 0, 0.05)' }}
              />
              <Legend />
              <Bar 
                dataKey="value" 
                name="Brainwave Power"
                fill={(entry) => entry.color} 
                radius={[4, 4, 0, 0]} 
                animationDuration={1000}
              />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={brainwaves}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={renderCustomizedLabel}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
                nameKey="name"
                animationDuration={1000}
              >
                {brainwaves.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => [`${value}%`, 'Power']} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="grid grid-cols-5 gap-2 mt-4">
        {brainwaves.map((wave) => (
          <Card key={wave.name} className="overflow-hidden">
            <CardContent className="p-3 pt-3">
              <div className="text-center">
                <p className="font-semibold" style={{ color: wave.color }}>{wave.name}</p>
                <p className="text-2xl font-bold">{wave.value}%</p>
                <p className="text-xs text-muted-foreground mt-1">{getBrainwaveDescription(wave.name)}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};

// Helper function to get brainwave descriptions
const getBrainwaveDescription = (waveName: string): string => {
  switch (waveName) {
    case 'Delta':
      return 'Deep sleep, healing';
    case 'Theta':
      return 'Creativity, meditation';
    case 'Alpha':
      return 'Relaxation, calmness';
    case 'Beta':
      return 'Active thinking, focus';
    case 'Gamma':
      return 'Higher processing, peak';
    default:
      return '';
  }
};

export default BrainwavesChart;