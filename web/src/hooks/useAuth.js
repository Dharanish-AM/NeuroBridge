import { useState, useEffect, createContext, useContext } from 'react';

const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const useAuthState = () => {
  const [authState, setAuthState] = useState({
    user: null,
    isAuthenticated: false,
    isLoading: true,
    error: null
  });

  useEffect(() => {
    // Check for existing session
    const savedUser = localStorage.getItem('neurobridge-user');
    if (savedUser) {
      try {
        const user = JSON.parse(savedUser);
        setAuthState({
          user,
          isAuthenticated: true,
          isLoading: false,
          error: null
        });
      } catch (error) {
        localStorage.removeItem('neurobridge-user');
        setAuthState(prev => ({ ...prev, isLoading: false }));
      }
    } else {
      setAuthState(prev => ({ ...prev, isLoading: false }));
    }
  }, []);

  const login = async (email, password) => {
    setAuthState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock user data - in real app, this would come from your backend
      const mockUser = {
        id: 'user_' + Date.now(),
        email,
        fullName: 'Demo User',
        age: 25,
        gender: 'prefer-not-to-say',
        country: 'United States',
        timezone: 'America/New_York',
        primaryGoals: ['Improve Focus'],
        userType: 'student',
        preferredTrainingTypes: ['Games', 'Meditation'],
        cognitiveLevel: 'beginner',
        hasEEGDevice: false,
        preferredSessionDuration: '10-20',
        dailyTrainingWindow: { from: '09:00', to: '17:00' },
        soundPreference: 'lo-fi',
        medicalConditions: [],
        medications: [],
        sleepHours: 8,
        stressLevel: 5,
        exerciseFrequency: 'weekly',
        learningStyle: 'visual',
        attentionSpan: 'medium',
        motivationFactors: ['Achievement', 'Progress'],
        language: 'English',
        theme: 'auto',
        enableSmartTips: true,
        allowNotifications: true,
        personalizedContent: true,
        dataSharing: false,
        consentGiven: true,
        privacyPolicyAccepted: true,
        marketingConsent: false,
        createdAt: new Date(),
        updatedAt: new Date(),
        lastLoginAt: new Date()
      };

      localStorage.setItem('neurobridge-user', JSON.stringify(mockUser));
      setAuthState({
        user: mockUser,
        isAuthenticated: true,
        isLoading: false,
        error: null
      });
      
      return true;
    } catch (error) {
      setAuthState(prev => ({
        ...prev,
        isLoading: false,
        error: 'Login failed. Please try again.'
      }));
      return false;
    }
  };

  const signup = async (userData, password) => {
    setAuthState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const newUser = {
        id: 'user_' + Date.now(),
        createdAt: new Date(),
        updatedAt: new Date(),
        lastLoginAt: new Date(),
        ...userData
      };

      localStorage.setItem('neurobridge-user', JSON.stringify(newUser));
      setAuthState({
        user: newUser,
        isAuthenticated: true,
        isLoading: false,
        error: null
      });
      
      return true;
    } catch (error) {
      setAuthState(prev => ({
        ...prev,
        isLoading: false,
        error: 'Signup failed. Please try again.'
      }));
      return false;
    }
  };

  const logout = () => {
    localStorage.removeItem('neurobridge-user');
    setAuthState({
      user: null,
      isAuthenticated: false,
      isLoading: false,
      error: null
    });
  };

  const updateProfile = async (updates) => {
    if (!authState.user) return false;
    
    try {
      const updatedUser = {
        ...authState.user,
        ...updates,
        updatedAt: new Date()
      };
      
      localStorage.setItem('neurobridge-user', JSON.stringify(updatedUser));
      setAuthState(prev => ({
        ...prev,
        user: updatedUser
      }));
      
      return true;
    } catch (error) {
      return false;
    }
  };

  return {
    authState,
    login,
    signup,
    logout,
    updateProfile
  };
};

export { AuthContext };