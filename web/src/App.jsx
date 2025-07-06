import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import AuthProvider from './components/Auth/AuthProvider';
import Layout from './components/Layout/Layout';
import Dashboard from './pages/Dashboard';
import Training from './pages/Training';
import Education from './pages/Education';
import Recommendations from './pages/Recommendations';
import Reports from './pages/Reports';
import FocusMode from './pages/FocusMode';
import Settings from './pages/Settings';
import Profile from './pages/Profile';

function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="App">
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<Dashboard />} />
              <Route path="train" element={<Training />} />
              <Route path="learn" element={<Education />} />
              <Route path="recommendations" element={<Recommendations />} />
              <Route path="reports" element={<Reports />} />
              <Route path="focus" element={<FocusMode />} />
              <Route path="settings" element={<Settings />} />
              <Route path="profile" element={<Profile />} />
            </Route>
          </Routes>
        </div>
      </Router>
    </AuthProvider>
  );
}

export default App;