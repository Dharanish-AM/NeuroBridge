import React from 'react';
import { useBrainData } from '@/context/BrainDataContext';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { BadgeDelta } from '@tremor/react';

// Mock data for daily stats
const mockDailyStats = [
  { name: 'Avg. Focus', value: 65, change: 12, changeType: 'increase' },
  { name: 'Deep Focus Sessions', value: 3, change: 1, changeType: 'increase' },
  { name: 'Relaxation Periods', value: 4, change: -1, changeType: 'decrease' },
  { name: 'Lesson Engagement', value: 78, change: 5, changeType: 'increase' },
];

const DailyStats: React.FC = () => {
  return (
    <div className="grid gap-4 grid-cols-1 md:grid-cols-2">
      {mockDailyStats.map((stat, index) => (
        <Card key={index} className="overflow-hidden">
          <CardContent className="p-4">
            <div className="flex justify-between items-center">
              <p className="text-sm font-medium text-muted-foreground">{stat.name}</p>
              <BadgeDelta
                deltaType={stat.changeType === 'increase' ? 'increase' : 'decrease'}
                isIncreasePositive={true}
                size="xs"
              >
                {stat.changeType === 'increase' ? '+' : ''}{stat.change}%
              </BadgeDelta>
            </div>
            {typeof stat.value === 'number' && stat.name.includes('Avg.') ? (
              <>
                <p className="text-2xl font-bold mt-2">{stat.value}%</p>
                <Progress value={stat.value} className="mt-2" />
              </>
            ) : typeof stat.value === 'number' && stat.name.includes('Engagement') ? (
              <>
                <p className="text-2xl font-bold mt-2">{stat.value}%</p>
                <Progress value={stat.value} className="mt-2" />
              </>
            ) : (
              <p className="text-2xl font-bold mt-2">{stat.value}</p>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

export default DailyStats;