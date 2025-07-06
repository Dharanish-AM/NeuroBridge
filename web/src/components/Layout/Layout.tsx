import React from 'react';
import { Outlet } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import Navbar from './Navbar';
import AuthModal from '../Auth/AuthModal';
import LandingPage from '../../pages/LandingPage';

const Layout: React.FC = () => {
  const { authState } = useAuth();

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
      <Navbar />
      <main>
        {authState.isAuthenticated ? (
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <Outlet />
          </div>
        ) : (
          <LandingPage />
        )}
      </main>
    </div>
  );
};

export default Layout;