import React, { createContext, useState, useContext, useEffect } from 'react';

// Define types for our brain data
export interface ChannelData {
  name: string;
  value: number;
  history: number[];
}

export interface BrainwaveData {
  name: string;
  value: number; // Percentage
  color: string;
}

export type MentalState = 'Focused' | 'Relaxed' | 'Drowsy' | 'Neutral';

interface BrainDataContextType {
  channels: ChannelData[];
  brainwaves: BrainwaveData[];
  mentalState: MentalState;
  focusLevel: number; // 0-100
  timeSeriesData: { time: number; [key: string]: number }[];
  updateBrainData: () => void;
}

// Create context with a default value
const BrainDataContext = createContext<BrainDataContextType | undefined>(undefined);

// Mock data generator function
const generateMockData = () => {
  // Generate random EEG channel data
  const channels: ChannelData[] = [
    { name: 'Fp1', value: Math.random() * 100 - 50, history: [] },
    { name: 'Fp2', value: Math.random() * 100 - 50, history: [] },
    { name: 'C3', value: Math.random() * 100 - 50, history: [] },
    { name: 'C4', value: Math.random() * 100 - 50, history: [] },
    { name: 'O1', value: Math.random() * 100 - 50, history: [] },
    { name: 'O2', value: Math.random() * 100 - 50, history: [] },
    { name: 'T3', value: Math.random() * 100 - 50, history: [] },
    { name: 'T4', value: Math.random() * 100 - 50, history: [] }
  ];

  // Generate random brainwave percentages (total = 100%)
  let remaining = 100;
  const brainwaves: BrainwaveData[] = [
    { name: 'Delta', value: 0, color: '#6366f1' }, // Indigo
    { name: 'Theta', value: 0, color: '#8b5cf6' }, // Purple
    { name: 'Alpha', value: 0, color: '#ec4899' }, // Pink
    { name: 'Beta', value: 0, color: '#14b8a6' },  // Teal
    { name: 'Gamma', value: 0, color: '#f97316' }  // Orange
  ];

  // Distribute the remaining percentage
  brainwaves.forEach((wave, index) => {
    if (index === brainwaves.length - 1) {
      wave.value = remaining;
    } else {
      const value = Math.floor(Math.random() * remaining * 0.7);
      wave.value = value;
      remaining -= value;
    }
  });

  // Determine mental state based on brainwave values
  const deltaValue = brainwaves.find(w => w.name === 'Delta')?.value || 0;
  const thetaValue = brainwaves.find(w => w.name === 'Theta')?.value || 0;
  const alphaValue = brainwaves.find(w => w.name === 'Alpha')?.value || 0;
  const betaValue = brainwaves.find(w => w.name === 'Beta')?.value || 0;

  let mentalState: MentalState = 'Neutral';
  if (betaValue > 30) {
    mentalState = 'Focused';
  } else if (alphaValue > 30) {
    mentalState = 'Relaxed';
  } else if (deltaValue + thetaValue > 60) {
    mentalState = 'Drowsy';
  }

  // Calculate focus level as a function of Beta/(Theta+Alpha)
  const focusLevel = Math.min(100, Math.max(0, 
    Math.round((betaValue / Math.max(1, thetaValue + alphaValue)) * 50)
  ));

  return {
    channels,
    brainwaves,
    mentalState,
    focusLevel
  };
};

export const BrainDataProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [channels, setChannels] = useState<ChannelData[]>([]);
  const [brainwaves, setBrainwaves] = useState<BrainwaveData[]>([]);
  const [mentalState, setMentalState] = useState<MentalState>('Neutral');
  const [focusLevel, setFocusLevel] = useState<number>(0);
  const [timeSeriesData, setTimeSeriesData] = useState<{ time: number; [key: string]: number }[]>([]);

  // Update brain data with new mock values
  const updateBrainData = () => {
    const { channels: newChannels, brainwaves: newBrainwaves, mentalState: newMentalState, focusLevel: newFocusLevel } = generateMockData();
    
    // Update channels and keep history
    const updatedChannels = newChannels.map((channel, index) => {
      const existingChannel = channels[index];
      const history = existingChannel ? 
        [...existingChannel.history.slice(-50), channel.value] : 
        [channel.value];
      
      return { ...channel, history };
    });

    // Update time series data
    const now = Date.now();
    const newTimePoint = {
      time: now,
      ...updatedChannels.reduce((acc, channel) => ({...acc, [channel.name]: channel.value}), {})
    };

    setChannels(updatedChannels);
    setBrainwaves(newBrainwaves);
    setMentalState(newMentalState);
    setFocusLevel(newFocusLevel);
    setTimeSeriesData(prev => [...prev.slice(-100), newTimePoint]);
  };

  // Initialize data on mount
  useEffect(() => {
    updateBrainData();
    // Set up interval for real-time updates
    const interval = setInterval(updateBrainData, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <BrainDataContext.Provider 
      value={{ 
        channels, 
        brainwaves, 
        mentalState, 
        focusLevel,
        timeSeriesData,
        updateBrainData
      }}
    >
      {children}
    </BrainDataContext.Provider>
  );
};

// Custom hook to use the brain data context
export const useBrainData = () => {
  const context = useContext(BrainDataContext);
  if (context === undefined) {
    throw new Error('useBrainData must be used within a BrainDataProvider');
  }
  return context;
};