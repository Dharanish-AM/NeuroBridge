import { Toaster } from '@/components/ui/sonner';
import { TooltipProvider } from '@/components/ui/tooltip';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useState } from 'react';

// Pages
import NotFound from './pages/NotFound';
import Dashboard from './pages/Dashboard';
import EEGSession from './pages/EEGSession';
import LessonRecommendations from './pages/LessonRecommendations';
import Profile from './pages/Profile';
import Layout from './components/layout/Layout';

// Context for brain data
import { BrainDataProvider } from './context/BrainDataContext';

const queryClient = new QueryClient();

const App = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <BrainDataProvider>
        <TooltipProvider>
          <Toaster />
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<Layout />}>
                <Route index element={<Navigate to="/dashboard" replace />} />
                <Route path="dashboard" element={<Dashboard />} />
                <Route path="eeg-session" element={<EEGSession />} />
                <Route path="lessons" element={<LessonRecommendations />} />
                <Route path="profile" element={<Profile />} />
              </Route>
              <Route path="*" element={<NotFound />} />
            </Routes>
          </BrowserRouter>
        </TooltipProvider>
      </BrainDataProvider>
    </QueryClientProvider>
  );
};

export default App;