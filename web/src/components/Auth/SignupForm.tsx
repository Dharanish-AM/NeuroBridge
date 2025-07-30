import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ArrowRight, 
  ArrowLeft, 
  Brain, 
  User, 
  Mail, 
  Lock, 
  Eye, 
  EyeOff,
  Target,
  Headphones,
  Settings,
  Shield,
  CheckCircle
} from 'lucide-react';
import { useAuth } from '../../hooks/useAuth';
import { UserProfile } from '../../types/user';

interface SignupFormProps {
  onSwitchToLogin: () => void;
  onClose?: () => void;
}

const SignupForm: React.FC<SignupFormProps> = ({ onSwitchToLogin, onClose }) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [formData, setFormData] = useState<Partial<UserProfile>>({
    primaryGoals: [],
    preferredTrainingTypes: [],
    medicalConditions: [],
    medications: [],
    motivationFactors: [],
    hasEEGDevice: false,
    enableSmartTips: true,
    allowNotifications: true,
    personalizedContent: true,
    dataSharing: false,
    consentGiven: false,
    privacyPolicyAccepted: false,
    marketingConsent: false
  });
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const { signup, authState } = useAuth();

  const totalSteps = 6;

  const validateStep = (step: number): boolean => {
    const newErrors: Record<string, string> = {};

    switch (step) {
      case 1:
        if (!formData.fullName) newErrors.fullName = 'Full name is required';
        if (!formData.email) newErrors.email = 'Email is required';
        else if (!/\S+@\S+\.\S+/.test(formData.email)) newErrors.email = 'Invalid email format';
        if (!password) newErrors.password = 'Password is required';
        else if (password.length < 6) newErrors.password = 'Password must be at least 6 characters';
        if (password !== confirmPassword) newErrors.confirmPassword = 'Passwords do not match';
        if (!formData.age || formData.age < 13) newErrors.age = 'Age must be 13 or older';
        break;
      case 2:
        if (!formData.primaryGoals || formData.primaryGoals.length === 0) {
          newErrors.primaryGoals = 'Please select at least one goal';
        }
        if (!formData.userType) newErrors.userType = 'Please select your user type';
        break;
      case 3:
        if (!formData.preferredSessionDuration) newErrors.preferredSessionDuration = 'Please select session duration';
        break;
      case 4:
        if (!formData.language) newErrors.language = 'Please select a language';
        if (!formData.theme) newErrors.theme = 'Please select a theme';
        break;
      case 5:
        if (!formData.consentGiven) newErrors.consent = 'You must agree to the terms';
        if (!formData.privacyPolicyAccepted) newErrors.privacy = 'You must accept the privacy policy';
        break;
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleNext = () => {
    if (validateStep(currentStep)) {
      setCurrentStep(prev => Math.min(prev + 1, totalSteps));
    }
  };

  const handlePrevious = () => {
    setCurrentStep(prev => Math.max(prev - 1, 1));
  };

  const handleSubmit = async () => {
    if (!validateStep(currentStep)) return;

    const success = await signup(formData, password);
    if (success && onClose) {
      onClose();
    }
  };

  const updateFormData = (field: string, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  const toggleArrayValue = (field: string, value: string) => {
    const currentArray = (formData[field as keyof UserProfile] as string[]) || [];
    const newArray = currentArray.includes(value)
      ? currentArray.filter(item => item !== value)
      : [...currentArray, value];
    updateFormData(field, newArray);
  };

  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return (
          <motion.div
            key="step1"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <div className="text-center mb-8">
              <User className="h-12 w-12 text-blue-600 mx-auto mb-4" />
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">Basic Information</h3>
              <p className="text-gray-600 dark:text-gray-400">Let's start with the essentials</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Full Name *
                </label>
                <input
                  type="text"
                  value={formData.fullName || ''}
                  onChange={(e) => updateFormData('fullName', e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  placeholder="Enter your full name"
                />
                {errors.fullName && <p className="mt-1 text-sm text-red-600">{errors.fullName}</p>}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Age *
                </label>
                <input
                  type="number"
                  value={formData.age || ''}
                  onChange={(e) => updateFormData('age', parseInt(e.target.value))}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  placeholder="Your age"
                  min="13"
                  max="120"
                />
                {errors.age && <p className="mt-1 text-sm text-red-600">{errors.age}</p>}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Email Address *
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                <input
                  type="email"
                  value={formData.email || ''}
                  onChange={(e) => updateFormData('email', e.target.value)}
                  className="w-full pl-10 pr-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  placeholder="Enter your email"
                />
              </div>
              {errors.email && <p className="mt-1 text-sm text-red-600">{errors.email}</p>}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Password *
                </label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full pl-10 pr-12 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    placeholder="Create password"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2"
                  >
                    {showPassword ? <EyeOff className="h-5 w-5 text-gray-400" /> : <Eye className="h-5 w-5 text-gray-400" />}
                  </button>
                </div>
                {errors.password && <p className="mt-1 text-sm text-red-600">{errors.password}</p>}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Confirm Password *
                </label>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  placeholder="Confirm password"
                />
                {errors.confirmPassword && <p className="mt-1 text-sm text-red-600">{errors.confirmPassword}</p>}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Gender
                </label>
                <select
                  value={formData.gender || ''}
                  onChange={(e) => updateFormData('gender', e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  <option value="">Select gender</option>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="non-binary">Non-binary</option>
                  <option value="prefer-not-to-say">Prefer not to say</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Country
                </label>
                <select
                  value={formData.country || ''}
                  onChange={(e) => updateFormData('country', e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  <option value="">Select country</option>
                  <option value="United States">United States</option>
                  <option value="Canada">Canada</option>
                  <option value="United Kingdom">United Kingdom</option>
                  <option value="India">India</option>
                  <option value="Australia">Australia</option>
                  <option value="Germany">Germany</option>
                  <option value="France">France</option>
                  <option value="Other">Other</option>
                </select>
              </div>
            </div>
          </motion.div>
        );

      case 2:
        return (
          <motion.div
            key="step2"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <div className="text-center mb-8">
              <Target className="h-12 w-12 text-blue-600 mx-auto mb-4" />
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">Your Goals</h3>
              <p className="text-gray-600 dark:text-gray-400">What do you want to achieve?</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
                Primary Goals * (Select all that apply)
              </label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {[
                  'Improve Focus',
                  'Reduce Anxiety',
                  'ADHD Training',
                  'Boost Memory',
                  'Cognitive Performance',
                  'General Relaxation',
                  'Better Sleep',
                  'Stress Management',
                  'Meditation Practice'
                ].map((goal) => (
                  <button
                    key={goal}
                    type="button"
                    onClick={() => toggleArrayValue('primaryGoals', goal)}
                    className={`p-3 text-sm rounded-lg border transition-colors ${
                      formData.primaryGoals?.includes(goal)
                        ? 'bg-blue-100 border-blue-300 text-blue-800 dark:bg-blue-900/20 dark:border-blue-600 dark:text-blue-300'
                        : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-600'
                    }`}
                  >
                    {goal}
                  </button>
                ))}
              </div>
              {errors.primaryGoals && <p className="mt-2 text-sm text-red-600">{errors.primaryGoals}</p>}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
                You are a: *
              </label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {[
                  { value: 'student', label: 'Student' },
                  { value: 'professional', label: 'Working Professional' },
                  { value: 'therapist', label: 'Therapist/Coach' },
                  { value: 'parent', label: 'Parent' },
                  { value: 'researcher', label: 'Researcher' },
                  { value: 'other', label: 'Other' }
                ].map((type) => (
                  <button
                    key={type.value}
                    type="button"
                    onClick={() => updateFormData('userType', type.value)}
                    className={`p-3 text-sm rounded-lg border transition-colors ${
                      formData.userType === type.value
                        ? 'bg-blue-100 border-blue-300 text-blue-800 dark:bg-blue-900/20 dark:border-blue-600 dark:text-blue-300'
                        : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-600'
                    }`}
                  >
                    {type.label}
                  </button>
                ))}
              </div>
              {errors.userType && <p className="mt-2 text-sm text-red-600">{errors.userType}</p>}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
                Preferred Training Types (Select all that apply)
              </label>
              <div className="grid grid-cols-2 gap-3">
                {[
                  'Breathing/Meditation',
                  'Games/Puzzles',
                  'Learning Videos',
                  'Study Support Tools',
                  'Biofeedback Training',
                  'Cognitive Exercises'
                ].map((type) => (
                  <button
                    key={type}
                    type="button"
                    onClick={() => toggleArrayValue('preferredTrainingTypes', type)}
                    className={`p-3 text-sm rounded-lg border transition-colors ${
                      formData.preferredTrainingTypes?.includes(type)
                        ? 'bg-blue-100 border-blue-300 text-blue-800 dark:bg-blue-900/20 dark:border-blue-600 dark:text-blue-300'
                        : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-600'
                    }`}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>
          </motion.div>
        );

      case 3:
        return (
          <motion.div
            key="step3"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <div className="text-center mb-8">
              <Headphones className="h-12 w-12 text-blue-600 mx-auto mb-4" />
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">EEG & Training Setup</h3>
              <p className="text-gray-600 dark:text-gray-400">Configure your training environment</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
                Do you have an EEG device?
              </label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => updateFormData('hasEEGDevice', true)}
                  className={`p-4 rounded-lg border transition-colors ${
                    formData.hasEEGDevice
                      ? 'bg-blue-100 border-blue-300 text-blue-800 dark:bg-blue-900/20 dark:border-blue-600 dark:text-blue-300'
                      : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300'
                  }`}
                >
                  Yes, I have one
                </button>
                <button
                  type="button"
                  onClick={() => updateFormData('hasEEGDevice', false)}
                  className={`p-4 rounded-lg border transition-colors ${
                    formData.hasEEGDevice === false
                      ? 'bg-blue-100 border-blue-300 text-blue-800 dark:bg-blue-900/20 dark:border-blue-600 dark:text-blue-300'
                      : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300'
                  }`}
                >
                  No, planning to get one
                </button>
              </div>
            </div>

            {formData.hasEEGDevice && (
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Select your EEG device:
                </label>
                <select
                  value={formData.eegDevice || ''}
                  onChange={(e) => updateFormData('eegDevice', e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  <option value="">Choose device</option>
                  <option value="Muse">Muse Headband</option>
                  <option value="Emotiv">Emotiv EPOC+</option>
                  <option value="NeuroSky">NeuroSky MindWave</option>
                  <option value="OpenBCI">OpenBCI</option>
                  <option value="Other">Other</option>
                </select>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
                Preferred session duration: *
              </label>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                  { value: '5-10', label: '5-10 min' },
                  { value: '10-20', label: '10-20 min' },
                  { value: '20-30', label: '20-30 min' },
                  { value: '30+', label: '30+ min' }
                ].map((duration) => (
                  <button
                    key={duration.value}
                    type="button"
                    onClick={() => updateFormData('preferredSessionDuration', duration.value)}
                    className={`p-3 text-sm rounded-lg border transition-colors ${
                      formData.preferredSessionDuration === duration.value
                        ? 'bg-blue-100 border-blue-300 text-blue-800 dark:bg-blue-900/20 dark:border-blue-600 dark:text-blue-300'
                        : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300'
                    }`}
                  >
                    {duration.label}
                  </button>
                ))}
              </div>
              {errors.preferredSessionDuration && <p className="mt-2 text-sm text-red-600">{errors.preferredSessionDuration}</p>}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Daily training window (From):
                </label>
                <input
                  type="time"
                  value={formData.dailyTrainingWindow?.from || '09:00'}
                  onChange={(e) => updateFormData('dailyTrainingWindow', {
                    ...formData.dailyTrainingWindow,
                    from: e.target.value
                  })}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Daily training window (To):
                </label>
                <input
                  type="time"
                  value={formData.dailyTrainingWindow?.to || '17:00'}
                  onChange={(e) => updateFormData('dailyTrainingWindow', {
                    ...formData.dailyTrainingWindow,
                    to: e.target.value
                  })}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
                Sound preference during sessions:
              </label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {[
                  { value: 'none', label: 'None' },
                  { value: 'lo-fi', label: 'Lo-Fi' },
                  { value: 'nature', label: 'Nature' },
                  { value: 'white-noise', label: 'White Noise' },
                  { value: 'binaural', label: 'Binaural Beats' }
                ].map((sound) => (
                  <button
                    key={sound.value}
                    type="button"
                    onClick={() => updateFormData('soundPreference', sound.value)}
                    className={`p-3 text-sm rounded-lg border transition-colors ${
                      formData.soundPreference === sound.value
                        ? 'bg-blue-100 border-blue-300 text-blue-800 dark:bg-blue-900/20 dark:border-blue-600 dark:text-blue-300'
                        : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300'
                    }`}
                  >
                    {sound.label}
                  </button>
                ))}
              </div>
            </div>
          </motion.div>
        );

      case 4:
        return (
          <motion.div
            key="step4"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <div className="text-center mb-8">
              <Settings className="h-12 w-12 text-blue-600 mx-auto mb-4" />
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">App Preferences</h3>
              <p className="text-gray-600 dark:text-gray-400">Customize your experience</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Language *
                </label>
                <select
                  value={formData.language || ''}
                  onChange={(e) => updateFormData('language', e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  <option value="">Select language</option>
                  <option value="English">English</option>
                  <option value="Spanish">Spanish</option>
                  <option value="French">French</option>
                  <option value="German">German</option>
                  <option value="Hindi">Hindi</option>
                  <option value="Tamil">Tamil</option>
                </select>
                {errors.language && <p className="mt-1 text-sm text-red-600">{errors.language}</p>}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Theme *
                </label>
                <select
                  value={formData.theme || ''}
                  onChange={(e) => updateFormData('theme', e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  <option value="">Select theme</option>
                  <option value="light">Light</option>
                  <option value="dark">Dark</option>
                  <option value="auto">Auto</option>
                </select>
                {errors.theme && <p className="mt-1 text-sm text-red-600">{errors.theme}</p>}
              </div>
            </div>

            <div className="space-y-4">
              <h4 className="text-lg font-medium text-gray-900 dark:text-white">Notification Preferences</h4>
              
              <div className="space-y-3">
                {[
                  { key: 'enableSmartTips', label: 'Enable Smart Tips', description: 'Get personalized recommendations based on your brain state' },
                  { key: 'allowNotifications', label: 'Training Reminders', description: 'Receive notifications for scheduled training sessions' },
                  { key: 'personalizedContent', label: 'Personalized Content', description: 'Show content adapted to your brainwave patterns' }
                ].map((setting) => (
                  <div key={setting.key} className="flex items-start space-x-3">
                    <input
                      type="checkbox"
                      checked={formData[setting.key as keyof UserProfile] as boolean || false}
                      onChange={(e) => updateFormData(setting.key, e.target.checked)}
                      className="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <div>
                      <label className="text-sm font-medium text-gray-900 dark:text-white">
                        {setting.label}
                      </label>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        {setting.description}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Health Information (Optional)</h4>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Average sleep hours per night
                  </label>
                  <input
                    type="number"
                    value={formData.sleepHours || ''}
                    onChange={(e) => updateFormData('sleepHours', parseInt(e.target.value))}
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    placeholder="8"
                    min="1"
                    max="12"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Current stress level (1-10)
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={formData.stressLevel || 5}
                    onChange={(e) => updateFormData('stressLevel', parseInt(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-center text-sm text-gray-600 dark:text-gray-400 mt-1">
                    {formData.stressLevel || 5}/10
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        );

      case 5:
        return (
          <motion.div
            key="step5"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <div className="text-center mb-8">
              <Shield className="h-12 w-12 text-blue-600 mx-auto mb-4" />
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">Privacy & Consent</h3>
              <p className="text-gray-600 dark:text-gray-400">Review and accept our terms</p>
            </div>

            <div className="space-y-6">
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 border border-blue-200 dark:border-blue-800">
                <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-3">
                  Data Usage & Privacy
                </h4>
                <p className="text-blue-800 dark:text-blue-200 text-sm mb-4">
                  NeuroBridge collects and processes your EEG data to provide personalized training recommendations. 
                  Your data is encrypted and stored securely. We never share personal information with third parties 
                  without your explicit consent.
                </p>
                <div className="space-y-3">
                  <label className="flex items-start space-x-3">
                    <input
                      type="checkbox"
                      checked={formData.consentGiven || false}
                      onChange={(e) => updateFormData('consentGiven', e.target.checked)}
                      className="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <span className="text-sm text-blue-800 dark:text-blue-200">
                      I agree to NeuroBridge's Terms of Service and understand how my data will be used *
                    </span>
                  </label>
                  {errors.consent && <p className="text-sm text-red-600 ml-7">{errors.consent}</p>}

                  <label className="flex items-start space-x-3">
                    <input
                      type="checkbox"
                      checked={formData.privacyPolicyAccepted || false}
                      onChange={(e) => updateFormData('privacyPolicyAccepted', e.target.checked)}
                      className="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <span className="text-sm text-blue-800 dark:text-blue-200">
                      I have read and accept the Privacy Policy *
                    </span>
                  </label>
                  {errors.privacy && <p className="text-sm text-red-600 ml-7">{errors.privacy}</p>}

                  <label className="flex items-start space-x-3">
                    <input
                      type="checkbox"
                      checked={formData.marketingConsent || false}
                      onChange={(e) => updateFormData('marketingConsent', e.target.checked)}
                      className="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <span className="text-sm text-blue-800 dark:text-blue-200">
                      I would like to receive updates about new features and research findings (optional)
                    </span>
                  </label>

                  <label className="flex items-start space-x-3">
                    <input
                      type="checkbox"
                      checked={formData.dataSharing || false}
                      onChange={(e) => updateFormData('dataSharing', e.target.checked)}
                      className="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <span className="text-sm text-blue-800 dark:text-blue-200">
                      Allow anonymous data contribution to neuroscience research (optional)
                    </span>
                  </label>
                </div>
              </div>
            </div>
          </motion.div>
        );

      case 6:
        return (
          <motion.div
            key="step6"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <div className="text-center mb-8">
              <CheckCircle className="h-12 w-12 text-green-600 mx-auto mb-4" />
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">Almost Done!</h3>
              <p className="text-gray-600 dark:text-gray-400">Review your information and create your account</p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6 space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium text-gray-900 dark:text-white">Name:</span>
                  <span className="ml-2 text-gray-600 dark:text-gray-400">{formData.fullName}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-900 dark:text-white">Email:</span>
                  <span className="ml-2 text-gray-600 dark:text-gray-400">{formData.email}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-900 dark:text-white">Primary Goals:</span>
                  <span className="ml-2 text-gray-600 dark:text-gray-400">
                    {formData.primaryGoals?.join(', ')}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-gray-900 dark:text-white">User Type:</span>
                  <span className="ml-2 text-gray-600 dark:text-gray-400">{formData.userType}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-900 dark:text-white">EEG Device:</span>
                  <span className="ml-2 text-gray-600 dark:text-gray-400">
                    {formData.hasEEGDevice ? formData.eegDevice || 'Yes' : 'Planning to get one'}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-gray-900 dark:text-white">Session Duration:</span>
                  <span className="ml-2 text-gray-600 dark:text-gray-400">{formData.preferredSessionDuration} minutes</span>
                </div>
              </div>
            </div>

            {authState.error && (
              <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                <p className="text-sm text-red-600 dark:text-red-400">{authState.error}</p>
              </div>
            )}
          </motion.div>
        );

      default:
        return null;
    }
  };

  return (
    <motion.div
      className="w-full max-w-4xl mx-auto"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="text-center mb-8">
        <div className="flex items-center justify-center mb-4">
          <div className="p-3 bg-blue-100 dark:bg-blue-900/20 rounded-full">
            <Brain className="h-8 w-8 text-blue-600 dark:text-blue-400" />
          </div>
        </div>
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
          Join NeuroBridge
        </h2>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Create your personalized brain training profile
        </p>
      </div>

      {/* Progress Bar */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            Step {currentStep} of {totalSteps}
          </span>
          <span className="text-sm text-gray-600 dark:text-gray-400">
            {Math.round((currentStep / totalSteps) * 100)}% Complete
          </span>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <motion.div
            className="bg-blue-500 h-2 rounded-full"
            animate={{ width: `${(currentStep / totalSteps) * 100}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </div>

      {/* Form Content */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <AnimatePresence mode="wait">
          {renderStep()}
        </AnimatePresence>

        {/* Navigation */}
        <div className="flex items-center justify-between pt-8 border-t border-gray-200 dark:border-gray-700 mt-8">
          <button
            type="button"
            onClick={currentStep === 1 ? onSwitchToLogin : handlePrevious}
            className="flex items-center space-x-2 px-6 py-3 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            <span>{currentStep === 1 ? 'Back to Login' : 'Previous'}</span>
          </button>

          {currentStep < totalSteps ? (
            <motion.button
              type="button"
              onClick={handleNext}
              className="flex items-center space-x-2 bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <span>Next</span>
              <ArrowRight className="h-4 w-4" />
            </motion.button>
          ) : (
            <motion.button
              type="button"
              onClick={handleSubmit}
              disabled={authState.isLoading}
              className="flex items-center space-x-2 bg-green-600 text-white px-8 py-3 rounded-lg font-medium hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              {authState.isLoading ? (
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <>
                  <span>Create Account</span>
                  <CheckCircle className="h-4 w-4" />
                </>
              )}
            </motion.button>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default SignupForm;