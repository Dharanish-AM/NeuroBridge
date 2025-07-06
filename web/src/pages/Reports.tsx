import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { BarChart3, TrendingUp, Calendar, Download, Filter } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, RadarChart as RechartsRadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

const Reports: React.FC = () => {
  const [timeRange, setTimeRange] = useState('7d');
  const [selectedMetric, setSelectedMetric] = useState('attention');

  // Generate sample historical data
  const generateHistoricalData = () => {
    const days = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : 90;
    return Array.from({ length: days }, (_, i) => ({
      date: new Date(Date.now() - (days - i) * 24 * 60 * 60 * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      attention: Math.random() * 0.4 + 0.4,
      meditation: Math.random() * 0.4 + 0.3,
      stress: Math.random() * 0.3 + 0.2,
      focus_sessions: Math.floor(Math.random() * 5) + 1,
      session_duration: Math.random() * 20 + 10
    }));
  };

  const performanceData = generateHistoricalData();

  const weeklyStats = [
    { day: 'Mon', focus: 85, meditation: 70, sessions: 3 },
    { day: 'Tue', focus: 78, meditation: 85, sessions: 2 },
    { day: 'Wed', focus: 92, meditation: 75, sessions: 4 },
    { day: 'Thu', focus: 76, meditation: 90, sessions: 2 },
    { day: 'Fri', focus: 88, meditation: 68, sessions: 3 },
    { day: 'Sat', focus: 95, meditation: 88, sessions: 5 },
    { day: 'Sun', focus: 82, meditation: 92, sessions: 3 }
  ];

  const skillsData = [
    { skill: 'Focus', current: 85, previous: 78, max: 100 },
    { skill: 'Meditation', current: 72, previous: 65, max: 100 },
    { skill: 'Memory', current: 68, previous: 62, max: 100 },
    { skill: 'Reaction', current: 74, previous: 71, max: 100 },
    { skill: 'Creativity', current: 81, previous: 75, max: 100 }
  ];

  const achievements = [
    { name: 'Focus Master', date: '2024-01-15', description: '10 consecutive focus sessions' },
    { name: 'Meditation Streak', date: '2024-01-10', description: '7-day meditation streak' },
    { name: 'Quick Learner', date: '2024-01-05', description: 'Completed 5 education modules' }
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Progress Reports
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Track your mental performance and training progress over time
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Filter className="h-4 w-4 text-gray-500" />
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 text-sm"
            >
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
              <option value="90d">Last 90 days</option>
            </select>
          </div>
          <button className="flex items-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
            <Download className="h-4 w-4" />
            <span>Export</span>
          </button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {[
          { label: 'Avg Attention', value: '82%', change: '+5%', trend: 'up', color: 'text-blue-600' },
          { label: 'Meditation Time', value: '45min', change: '+12min', trend: 'up', color: 'text-green-600' },
          { label: 'Training Sessions', value: '24', change: '+8', trend: 'up', color: 'text-purple-600' },
          { label: 'Stress Level', value: '18%', change: '-7%', trend: 'down', color: 'text-red-600' }
        ].map((metric, index) => (
          <motion.div
            key={metric.label}
            className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">{metric.label}</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">{metric.value}</p>
                <p className={`text-sm ${metric.trend === 'up' ? 'text-green-600' : 'text-red-600'} mt-1`}>
                  {metric.change} vs last period
                </p>
              </div>
              <TrendingUp className={`h-8 w-8 ${metric.color}`} />
            </div>
          </motion.div>
        ))}
      </div>

      {/* Performance Trends */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <motion.div
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Performance Trends
            </h3>
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
              className="bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1 text-sm"
            >
              <option value="attention">Attention</option>
              <option value="meditation">Meditation</option>
              <option value="stress">Stress</option>
            </select>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis dataKey="date" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey={selectedMetric}
                stroke="#3B82F6"
                strokeWidth={3}
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>

        <motion.div
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
            Weekly Activity
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={weeklyStats}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis dataKey="day" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="sessions" fill="#3B82F6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Skills Development */}
      <motion.div
        className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
          Skills Development
        </h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="space-y-4">
            {skillsData.map((skill, index) => (
              <div key={skill.skill} className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-gray-900 dark:text-white">{skill.skill}</span>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {skill.current}% (+{skill.current - skill.previous})
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <motion.div
                    className="bg-blue-500 h-2 rounded-full"
                    initial={{ width: `${skill.previous}%` }}
                    animate={{ width: `${skill.current}%` }}
                    transition={{ duration: 1, delay: index * 0.1 }}
                  />
                </div>
              </div>
            ))}
          </div>
          <div className="flex items-center justify-center">
            <ResponsiveContainer width="100%" height={250}>
              <RechartsRadarChart data={skillsData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="skill" />
                <PolarRadiusAxis angle={90} domain={[0, 100]} tick={false} axisLine={false} />
                <Radar
                  name="Current"
                  dataKey="current"
                  stroke="#3B82F6"
                  fill="#3B82F6"
                  fillOpacity={0.2}
                  strokeWidth={2}
                />
                <Radar
                  name="Previous"
                  dataKey="previous"
                  stroke="#94A3B8"
                  fill="transparent"
                  strokeWidth={1}
                  strokeDasharray="5 5"
                />
              </RechartsRadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </motion.div>

      {/* Recent Achievements */}
      <motion.div
        className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.5 }}
      >
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
          Recent Achievements
        </h3>
        <div className="space-y-4">
          {achievements.map((achievement, index) => (
            <motion.div
              key={achievement.name}
              className="flex items-center space-x-4 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.6 + index * 0.1 }}
            >
              <div className="p-2 bg-yellow-500 rounded-full">
                <BarChart3 className="h-5 w-5 text-white" />
              </div>
              <div className="flex-1">
                <h4 className="font-semibold text-yellow-900 dark:text-yellow-100">
                  {achievement.name}
                </h4>
                <p className="text-sm text-yellow-700 dark:text-yellow-300">
                  {achievement.description}
                </p>
              </div>
              <span className="text-sm text-yellow-600 dark:text-yellow-400">
                {achievement.date}
              </span>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </div>
  );
};

export default Reports;