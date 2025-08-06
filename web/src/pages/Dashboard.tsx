import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useBrainData } from '@/context/BrainDataContext';
import BrainwavesChart from '@/components/dashboard/BrainwavesChart';
import FocusGauge from '@/components/dashboard/FocusGauge';
import RecentSessions from '@/components/dashboard/RecentSessions';
import DailyStats from '@/components/dashboard/DailyStats';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { BarChart3, BookOpen, Brain, CalendarDays, Clock, LineChart, Zap } from 'lucide-react';
import { Link } from 'react-router-dom';
import BrainActivityMap from '@/components/dashboard/BrainActivityMap';


type TimeRangeType = 'today' | 'week' | 'month';

const Dashboard: React.FC = () => {
  const { mentalState, focusLevel, brainwaves } = useBrainData();
  const [timeRange, setTimeRange] = useState<TimeRangeType>('today');

  
  const getMentalStateBadgeVariant = (state: string) => {
    switch (state) {
      case 'Focused': return 'default'; 
      case 'Relaxed': return 'secondary'; 
      case 'Drowsy': return 'destructive'; 
      default: return 'outline';
    }
  };

  
  const getDominantBrainwave = () => {
    if (!brainwaves.length) return { name: 'None', value: 0 };
    return brainwaves.reduce((prev, current) => (current.value > prev.value) ? current : prev);
  };

  const dominantWave = getDominantBrainwave();

  return (
    <div className="container mx-auto space-y-8">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Dashboard</h2>
          <p className="text-muted-foreground">Welcome back. Monitor your cognitive performance in real-time.</p>
        </div>
        <div className="flex flex-col sm:flex-row items-end sm:items-center gap-2">
          <Badge variant={getMentalStateBadgeVariant(mentalState)} className="px-3 py-1">
            Current Mental State: {mentalState}
          </Badge>
          <Button asChild size="sm" variant="outline">
            <Link to="/eeg-session">
              <Zap className="mr-1 h-4 w-4" />
              Start New Session
            </Link>
          </Button>
        </div>
      </div>

      {/* Main Stats Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <Card className="overflow-hidden hover:shadow-md transition-all">
          <CardHeader className="pb-2 bg-gradient-to-r from-blue-50 to-indigo-50">
            <div className="flex justify-between items-center">
              <CardTitle className="text-sm font-medium text-muted-foreground">Focus Level</CardTitle>
              <div className="bg-white p-1.5 rounded-full shadow-sm">
                <Brain className="h-4 w-4 text-blue-500" />
              </div>
            </div>
          </CardHeader>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">{focusLevel}%</div>
            <Progress value={focusLevel} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-2 flex items-center gap-1">
              <span className={`h-2 w-2 rounded-full ${focusLevel > 70 ? 'bg-green-500' : focusLevel > 50 ? 'bg-amber-500' : 'bg-red-500'}`}></span>
              {focusLevel > 70 ? 'High Focus' : focusLevel > 50 ? 'Medium Focus' : 'Low Focus'}
            </p>
          </CardContent>
        </Card>
        
        <Card className="overflow-hidden hover:shadow-md transition-all">
          <CardHeader className="pb-2 bg-gradient-to-r from-purple-50 to-pink-50">
            <div className="flex justify-between items-center">
              <CardTitle className="text-sm font-medium text-muted-foreground">Dominant Brainwave</CardTitle>
              <div className="bg-white p-1.5 rounded-full shadow-sm">
                <LineChart className="h-4 w-4 text-purple-500" />
              </div>
            </div>
          </CardHeader>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">{dominantWave.name}</div>
            <div className="h-2 rounded-full mt-2" style={{ backgroundColor: dominantWave.color || '#cbd5e1', width: `${dominantWave.value}%` }}></div>
            <p className="text-xs text-muted-foreground mt-2 flex items-center gap-1">
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: dominantWave.color || '#cbd5e1' }}></span>
              {dominantWave.value}% power - {getBrainwaveDescription(dominantWave.name)}
            </p>
          </CardContent>
        </Card>
        
        <Card className="overflow-hidden hover:shadow-md transition-all">
          <CardHeader className="pb-2 bg-gradient-to-r from-green-50 to-emerald-50">
            <div className="flex justify-between items-center">
              <CardTitle className="text-sm font-medium text-muted-foreground">Learning Progress</CardTitle>
              <div className="bg-white p-1.5 rounded-full shadow-sm">
                <BookOpen className="h-4 w-4 text-green-500" />
              </div>
            </div>
          </CardHeader>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">3 of 5</div>
            <Progress value={60} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-2 flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-green-500"></span>
              60% complete - 2 lessons remaining
            </p>
          </CardContent>
        </Card>
        
        <Card className="overflow-hidden hover:shadow-md transition-all">
          <CardHeader className="pb-2 bg-gradient-to-r from-amber-50 to-orange-50">
            <div className="flex justify-between items-center">
              <CardTitle className="text-sm font-medium text-muted-foreground">Learning Streak</CardTitle>
              <div className="bg-white p-1.5 rounded-full shadow-sm">
                <CalendarDays className="h-4 w-4 text-amber-500" />
              </div>
            </div>
          </CardHeader>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">7 days</div>
            <div className="flex gap-1 mt-2">
              {Array.from({ length: 7 }).map((_, i) => (
                <div 
                  key={i} 
                  className="h-2 flex-1 rounded-sm bg-amber-400"
                  style={{ opacity: 0.5 + ((i + 1) / 14) }}
                ></div>
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-2 flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-amber-500"></span>
              Keep learning daily - +2 days this week
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Focus Gauge and Brainwaves Chart */}
      <div className="grid gap-6 md:grid-cols-2">
        <Card className="col-span-1 overflow-hidden hover:shadow-md transition-all">
          <CardHeader className="bg-gradient-to-r from-slate-50 to-blue-50">
            <CardTitle>Focus Gauge</CardTitle>
            <CardDescription>Real-time focus level indicator</CardDescription>
          </CardHeader>
          <CardContent className="flex justify-center py-6">
            <FocusGauge value={focusLevel} />
          </CardContent>
        </Card>
        
        <Card className="col-span-1 overflow-hidden hover:shadow-md transition-all">
          <CardHeader className="bg-gradient-to-r from-slate-50 to-indigo-50">
            <div className="flex justify-between">
              <div>
                <CardTitle>Brainwave Activity</CardTitle>
                <CardDescription>Distribution of brain wave frequencies</CardDescription>
              </div>
              <div>
                <Tabs defaultValue="today" value={timeRange} onValueChange={(val: string) => setTimeRange(val as TimeRangeType)}>
                  <TabsList className="grid grid-cols-3 h-8">
                    <TabsTrigger value="today" className="text-xs">Today</TabsTrigger>
                    <TabsTrigger value="week" className="text-xs">Week</TabsTrigger>
                    <TabsTrigger value="month" className="text-xs">Month</TabsTrigger>
                  </TabsList>
                </Tabs>
              </div>
            </div>
          </CardHeader>
          <CardContent className="py-6">
            <BrainwavesChart />
          </CardContent>
        </Card>
      </div>

      {/* Brain Activity Visualization */}
      <Card className="overflow-hidden hover:shadow-md transition-all">
        <CardHeader className="bg-gradient-to-r from-slate-50 to-purple-50">
          <CardTitle>Brain Activity Map</CardTitle>
          <CardDescription>Real-time EEG activity across brain regions</CardDescription>
        </CardHeader>
        <CardContent className="py-6">
          <BrainActivityMap />
        </CardContent>
      </Card>

      {/* Tabbed Analytics */}
      <Tabs defaultValue="daily" className="w-full">
        <TabsList>
          <TabsTrigger value="daily">Daily Stats</TabsTrigger>
          <TabsTrigger value="sessions">Recent Sessions</TabsTrigger>
        </TabsList>
        <TabsContent value="daily">
          <Card className="overflow-hidden hover:shadow-md transition-all">
            <CardHeader className="bg-gradient-to-r from-slate-50 to-blue-50">
              <CardTitle>Today's Progress</CardTitle>
              <CardDescription>
                Your learning metrics for today
              </CardDescription>
            </CardHeader>
            <CardContent>
              <DailyStats />
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="sessions">
          <Card className="overflow-hidden hover:shadow-md transition-all">
            <CardHeader className="bg-gradient-to-r from-slate-50 to-green-50">
              <CardTitle>Recent Learning Sessions</CardTitle>
              <CardDescription>
                Your latest EEG monitoring sessions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <RecentSessions />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};


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

export default Dashboard;