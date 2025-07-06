import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Brain, 
  Activity, 
  Target, 
  BookOpen, 
  Lightbulb, 
  BarChart3, 
  Focus, 
  Settings,
  Moon,
  Sun,
  User,
  LogOut,
  Menu,
  X
} from 'lucide-react';
import { useTheme } from '../../hooks/useTheme';
import { useAuth } from '../../hooks/useAuth';
import AuthModal from '../Auth/AuthModal';

const Navbar: React.FC = () => {
  const location = useLocation();
  const { theme, toggleTheme } = useTheme();
  const { authState, logout } = useAuth();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'signup'>('login');
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showMobileMenu, setShowMobileMenu] = useState(false);

  const navItems = [
    { path: '/', icon: Activity, label: 'Live Monitor' },
    { path: '/train', icon: Target, label: 'Train Mind' },
    { path: '/learn', icon: BookOpen, label: 'Learn Brain' },
    { path: '/recommendations', icon: Lightbulb, label: 'Recommendations' },
    { path: '/reports', icon: BarChart3, label: 'Reports' },
    { path: '/focus', icon: Focus, label: 'Focus Mode' },
    { path: '/settings', icon: Settings, label: 'Settings' },
  ];

  const handleAuthClick = (mode: 'login' | 'signup') => {
    setAuthMode(mode);
    setShowAuthModal(true);
  };

  const handleLogout = () => {
    logout();
    setShowUserMenu(false);
  };

  return (
    <>
      <nav className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-lg border-b border-gray-200 dark:border-gray-700 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link to="/" className="flex items-center space-x-3">
              <div className="relative">
                <Brain className="h-8 w-8 text-blue-600 dark:text-blue-400" />
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              </div>
              <span className="text-xl font-bold text-gray-900 dark:text-white">
                NeuroBridge
              </span>
            </Link>

            {/* Desktop Navigation */}
            {authState.isAuthenticated && (
              <div className="hidden lg:flex items-center space-x-1">
                {navItems.map((item) => {
                  const Icon = item.icon;
                  const isActive = location.pathname === item.path;
                  
                  return (
                    <Link key={item.path} to={item.path} className="relative">
                      <motion.div
                        className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                          isActive
                            ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20'
                            : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-50 dark:hover:bg-gray-800'
                        }`}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        <Icon className="h-4 w-4" />
                        <span className="hidden xl:block">{item.label}</span>
                      </motion.div>
                      {isActive && (
                        <motion.div
                          className="absolute bottom-0 left-1/2 w-1 h-1 bg-blue-600 dark:bg-blue-400 rounded-full"
                          layoutId="activeIndicator"
                          initial={false}
                          style={{ x: '-50%' }}
                        />
                      )}
                    </Link>
                  );
                })}
              </div>
            )}

            <div className="flex items-center space-x-4">
              <button
                onClick={toggleTheme}
                className="p-2 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              >
                {theme === 'light' ? (
                  <Moon className="h-5 w-5" />
                ) : (
                  <Sun className="h-5 w-5" />
                )}
              </button>

              {authState.isAuthenticated ? (
                <>
                  {/* Mobile Menu Button */}
                  <button
                    onClick={() => setShowMobileMenu(!showMobileMenu)}
                    className="lg:hidden p-2 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                  >
                    {showMobileMenu ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
                  </button>

                  {/* User Menu */}
                  <div className="relative">
                    <button
                      onClick={() => setShowUserMenu(!showUserMenu)}
                      className="flex items-center space-x-2 p-2 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                    >
                      <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                        <User className="h-4 w-4 text-white" />
                      </div>
                      <span className="hidden md:block text-sm font-medium text-gray-900 dark:text-white">
                        {authState.user?.fullName?.split(' ')[0] || 'User'}
                      </span>
                    </button>

                    {showUserMenu && (
                      <motion.div
                        className="absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 py-1 overflow-hidden"
                        initial={{ opacity: 0, y: -10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -10, scale: 0.95 }}
                        transition={{ duration: 0.2 }}
                      >
                        <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
                          <p className="text-sm font-medium text-gray-900 dark:text-white">
                            {authState.user?.fullName}
                          </p>
                          <p className="text-xs text-gray-600 dark:text-gray-400 truncate">
                            {authState.user?.email}
                          </p>
                        </div>
                        <Link
                          to="/profile"
                          className="flex items-center space-x-2 px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                          onClick={() => setShowUserMenu(false)}
                        >
                          <User className="h-4 w-4" />
                          <span>Profile</span>
                        </Link>
                        <Link
                          to="/settings"
                          className="flex items-center space-x-2 px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                          onClick={() => setShowUserMenu(false)}
                        >
                          <Settings className="h-4 w-4" />
                          <span>Settings</span>
                        </Link>
                        <button
                          onClick={handleLogout}
                          className="flex items-center space-x-2 w-full px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                        >
                          <LogOut className="h-4 w-4" />
                          <span>Sign Out</span>
                        </button>
                      </motion.div>
                    )}
                  </div>
                </>
              ) : (
                <div className="flex items-center space-x-3">
                  <button
                    onClick={() => handleAuthClick('login')}
                    className="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white font-medium transition-colors"
                  >
                    Sign In
                  </button>
                  <button
                    onClick={() => handleAuthClick('signup')}
                    className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-4 py-2 rounded-lg font-medium hover:from-blue-700 hover:to-purple-700 transition-all duration-300 shadow-lg hover:shadow-xl"
                  >
                    Get Started
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Mobile Menu */}
          {authState.isAuthenticated && showMobileMenu && (
            <motion.div
              className="lg:hidden border-t border-gray-200 dark:border-gray-700 py-4"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div className="space-y-2">
                {navItems.map((item) => {
                  const Icon = item.icon;
                  const isActive = location.pathname === item.path;
                  
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                        isActive
                          ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20'
                          : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-50 dark:hover:bg-gray-800'
                      }`}
                      onClick={() => setShowMobileMenu(false)}
                    >
                      <Icon className="h-5 w-5" />
                      <span className="font-medium">{item.label}</span>
                    </Link>
                  );
                })}
              </div>
            </motion.div>
          )}
        </div>
      </nav>

      <AuthModal
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        initialMode={authMode}
      />
    </>
  );
};

export default Navbar;