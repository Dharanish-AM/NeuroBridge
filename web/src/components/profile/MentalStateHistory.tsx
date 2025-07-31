import React from 'react';
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from 'recharts';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

// Mock data for mental state history
const generateMockHistoryData = () => {
  const states = ['Focused', 'Relaxed', 'Drowsy', 'Neutral'];
  const today = new Date();
  
  // Daily data (last 7 days)
  const dailyData = Array.from({ length: 7 }, (_, i) => {
    const date = new Date(today);
    date.setDate(date.getDate() - (6 - i));
    return {
      date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      focused: Math.floor(Math.random() * 40) + 30, // 30-70%
      relaxed: Math.floor(Math.random() * 30) + 10, // 10-40%
      drowsy: Math.floor(Math.random() * 20),       // 0-20%
      neutral: Math.floor(Math.random() * 10)       // 0-10%
    };
  });
  
  // Weekly data (last 4 weeks)
  const weeklyData = Array.from({ length: 4 }, (_, i) => {
    const date = new Date(today);
    date.setDate(date.getDate() - ((3 - i) * 7));
    return {
      date: `Week ${i + 1}`,
      focused: Math.floor(Math.random() * 40) + 30,
      relaxed: Math.floor(Math.random() * 30) + 10,
      drowsy: Math.floor(Math.random() * 20),
      neutral: Math.floor(Math.random() * 10)
    };
  });
  
  // Monthly data (last 6 months)
  const monthlyData = Array.from({ length: 6 }, (_, i) => {
    const date = new Date(today);
    date.setMonth(date.getMonth() - (5 - i));
    return {
      date: date.toLocaleDateString('en-US', { month: 'short' }),
      focused: Math.floor(Math.random() * 40) + 30,
      relaxed: Math.floor(Math.random() * 30) + 10,
      drowsy: Math.floor(Math.random() * 20),
      neutral: Math.floor(Math.random() * 10)
    };
  });
  
  return { dailyData, weeklyData, monthlyData };
};

const { dailyData, weeklyData, monthlyData } = generateMockHistoryData();

const MentalStateHistory: React.FC = () => {
  return (
    <Tabs defaultValue="daily">
      <TabsList className="mb-4">
        <TabsTrigger value="daily">Daily</TabsTrigger>
        <TabsTrigger value="weekly">Weekly</TabsTrigger>
        <TabsTrigger value="monthly">Monthly</TabsTrigger>
      </TabsList>
      
      <TabsContent value="daily">
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={dailyData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis dataKey="date" />
              <YAxis label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Line type="monotone" dataKey="focused" name="Focused" stroke="#14b8a6" strokeWidth={2} />
              <Line type="monotone" dataKey="relaxed" name="Relaxed" stroke="#8b5cf6" strokeWidth={2} />
              <Line type="monotone" dataKey="drowsy" name="Drowsy" stroke="#f97316" strokeWidth={2} />
              <Line type="monotone" dataKey="neutral" name="Neutral" stroke="#94a3b8" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </TabsContent>
      
      <TabsContent value="weekly">
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={weeklyData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis dataKey="date" />
              <YAxis label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Line type="monotone" dataKey="focused" name="Focused" stroke="#14b8a6" strokeWidth={2} />
              <Line type="monotone" dataKey="relaxed" name="Relaxed" stroke="#8b5cf6" strokeWidth={2} />
              <Line type="monotone" dataKey="drowsy" name="Drowsy" stroke="#f97316" strokeWidth={2} />
              <Line type="monotone" dataKey="neutral" name="Neutral" stroke="#94a3b8" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </TabsContent>
      
      <TabsContent value="monthly">
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={monthlyData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis dataKey="date" />
              <YAxis label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Line type="monotone" dataKey="focused" name="Focused" stroke="#14b8a6" strokeWidth={2} />
              <Line type="monotone" dataKey="relaxed" name="Relaxed" stroke="#8b5cf6" strokeWidth={2} />
              <Line type="monotone" dataKey="drowsy" name="Drowsy" stroke="#f97316" strokeWidth={2} />
              <Line type="monotone" dataKey="neutral" name="Neutral" stroke="#94a3b8" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </TabsContent>
    </Tabs>
  );
};

export default MentalStateHistory;