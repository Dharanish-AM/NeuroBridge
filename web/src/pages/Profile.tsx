import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  User, 
  Edit3, 
  Save, 
  X, 
  Camera, 
  Mail, 
  Phone, 
  MapPin, 
  Calendar, 
  Target, 
  Brain, 
  Activity, 
  Settings, 
  Shield, 
  Award, 
  TrendingUp,
  Clock,
  Headphones,
  Heart,
  Zap,
  BookOpen,
  BarChart3,
  Globe,
  Bell,
  Eye,
  Lock,
  Download,
  Upload,
  Trash2,
  CheckCircle,
  AlertCircle,
  Info
} from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { UserProfile } from '../types/user';

const Profile: React.FC = () => {
  const { authState, updateProfile } = useAuth();
  const [isEditing, setIsEditing] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [editData, setEditData] = useState<Partial<UserProfile>>(authState.user || {});
  const [isLoading, setIsLoading] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const user = authState.user;
  if (!user) return null;

  const tabs = [
    { id: 'overview', label: 'Overview', icon: User },
    { id: 'goals', label: 'Goals & Training', icon: Target },
    { id: 'health', label: 'Health & Wellness', icon: Heart },
    { id: 'devices', label: 'Devices & Setup', icon: Headphones },
    { id: 'preferences', label: 'Preferences', icon: Settings },
    { id: 'privacy', label: 'Privacy & Security', icon: Shield },
    { id: 'achievements', label: 'Achievements', icon: Award }
  ];

  const handleSave = async () => {
    setIsLoading(true);
    try {
      await updateProfile(editData);
      setIsEditing(false);
    } catch (error) {
      console.error('Failed to update profile:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancel = () => {
    setEditData(user);
    setIsEditing(false);
  };

  const updateEditData = (field: string, value: any) => {
    setEditData(prev => ({ ...prev, [field]: value }));
  };

  const toggleArrayValue = (field: string, value: string) => {
    const currentArray = (editData[field as keyof UserProfile] as string[]) || [];
    const newArray = currentArray.includes(value)
      ? currentArray.filter(item => item !== value)
      : [...currentArray, value];
    updateEditData(field, newArray);
  };

  const renderOverviewTab = () => (
    <div className="space-y-8">
      {/* Profile Header */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-8">
        <div className="flex flex-col md:flex-row items-center space-y-6 md:space-y-0 md:space-x-8">
          <div className="relative">
            <div className="w-32 h-32 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white text-4xl font-bold">
              {user.fullName?.charAt(0) || 'U'}
            </div>
            {isEditing && (
              <button className="absolute bottom-0 right-0 w-10 h-10 bg-blue-600 text-white rounded-full flex items-center justify-center hover:bg-blue-700 transition-colors">
                <Camera className="h-5 w-5" />
              </button>
            )}
          </div>
          
          <div className="flex-1 text-center md:text-left">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              {user.fullName}
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              {user.userType?.charAt(0).toUpperCase() + user.userType?.slice(1)} • Member since {new Date(user.createdAt).getFullYear()}
            </p>
            <div className="flex flex-wrap items-center justify-center md:justify-start gap-4 text-sm text-gray-600 dark:text-gray-400">
              <div className="flex items-center space-x-2">
                <MapPin className="h-4 w-4" />
                <span>{user.country}</span>
              </div>
              <div className="flex items-center space-x-2">
                <Calendar className="h-4 w-4" />
                <span>{user.age} years old</span>
              </div>
              <div className="flex items-center space-x-2">
                <Globe className="h-4 w-4" />
                <span>{user.timezone}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {[
          { label: 'Training Sessions', value: '127', icon: Target, color: 'blue' },
          { label: 'Total Hours', value: '42.5', icon: Clock, color: 'green' },
          { label: 'Avg Focus Score', value: '85%', icon: Brain, color: 'purple' },
          { label: 'Streak Days', value: '12', icon: TrendingUp, color: 'orange' }
        ].map((stat, index) => {
          const Icon = stat.icon;
          return (
            <motion.div
              key={index}
              className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700"
              whileHover={{ scale: 1.02 }}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{stat.label}</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">{stat.value}</p>
                </div>
                <div className={`p-3 rounded-lg bg-${stat.color}-100 dark:bg-${stat.color}-900/20`}>
                  <Icon className={`h-6 w-6 text-${stat.color}-600 dark:text-${stat.color}-400`} />
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Basic Information */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Basic Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Full Name
            </label>
            {isEditing ? (
              <input
                type="text"
                value={editData.fullName || ''}
                onChange={(e) => updateEditData('fullName', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              />
            ) : (
              <p className="text-gray-900 dark:text-white">{user.fullName}</p>
            )}
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Email Address
            </label>
            <div className="flex items-center space-x-2">
              <Mail className="h-4 w-4 text-gray-400" />
              <p className="text-gray-900 dark:text-white">{user.email}</p>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Age
            </label>
            {isEditing ? (
              <input
                type="number"
                value={editData.age || ''}
                onChange={(e) => updateEditData('age', parseInt(e.target.value))}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              />
            ) : (
              <p className="text-gray-900 dark:text-white">{user.age} years old</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Gender
            </label>
            {isEditing ? (
              <select
                value={editData.gender || ''}
                onChange={(e) => updateEditData('gender', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="non-binary">Non-binary</option>
                <option value="prefer-not-to-say">Prefer not to say</option>
              </select>
            ) : (
              <p className="text-gray-900 dark:text-white capitalize">{user.gender?.replace('-', ' ')}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Country
            </label>
            {isEditing ? (
              <select
                value={editData.country || ''}
                onChange={(e) => updateEditData('country', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                <option value="United States">United States</option>
                <option value="Canada">Canada</option>
                <option value="United Kingdom">United Kingdom</option>
                <option value="India">India</option>
                <option value="Australia">Australia</option>
                <option value="Germany">Germany</option>
                <option value="France">France</option>
                <option value="Other">Other</option>
              </select>
            ) : (
              <p className="text-gray-900 dark:text-white">{user.country}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Timezone
            </label>
            {isEditing ? (
              <input
                type="text"
                value={editData.timezone || ''}
                onChange={(e) => updateEditData('timezone', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              />
            ) : (
              <p className="text-gray-900 dark:text-white">{user.timezone}</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  const renderGoalsTab = () => (
    <div className="space-y-8">
      {/* Primary Goals */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Primary Goals</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {[
            'Improve Focus', 'Reduce Anxiety', 'ADHD Training', 'Boost Memory',
            'Cognitive Performance', 'General Relaxation', 'Better Sleep',
            'Stress Management', 'Meditation Practice'
          ].map((goal) => (
            <button
              key={goal}
              onClick={() => isEditing && toggleArrayValue('primaryGoals', goal)}
              disabled={!isEditing}
              className={`p-3 text-sm rounded-lg border transition-colors ${
                (editData.primaryGoals || user.primaryGoals)?.includes(goal)
                  ? 'bg-blue-100 border-blue-300 text-blue-800 dark:bg-blue-900/20 dark:border-blue-600 dark:text-blue-300'
                  : 'bg-gray-50 border-gray-300 text-gray-700 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300'
              } ${isEditing ? 'hover:bg-gray-100 dark:hover:bg-gray-600 cursor-pointer' : 'cursor-default'}`}
            >
              {goal}
            </button>
          ))}
        </div>
      </div>

      {/* User Type & Training Preferences */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">User Type</h3>
          <div className="space-y-3">
            {[
              { value: 'student', label: 'Student' },
              { value: 'professional', label: 'Working Professional' },
              { value: 'therapist', label: 'Therapist/Coach' },
              { value: 'parent', label: 'Parent' },
              { value: 'researcher', label: 'Researcher' },
              { value: 'other', label: 'Other' }
            ].map((type) => (
              <label key={type.value} className="flex items-center space-x-3">
                <input
                  type="radio"
                  name="userType"
                  value={type.value}
                  checked={(editData.userType || user.userType) === type.value}
                  onChange={(e) => isEditing && updateEditData('userType', e.target.value)}
                  disabled={!isEditing}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                />
                <span className="text-gray-900 dark:text-white">{type.label}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Cognitive Level</h3>
          <div className="space-y-3">
            {[
              { value: 'beginner', label: 'Beginner', desc: 'New to brain training' },
              { value: 'intermediate', label: 'Intermediate', desc: 'Some experience' },
              { value: 'advanced', label: 'Advanced', desc: 'Experienced user' }
            ].map((level) => (
              <label key={level.value} className="flex items-start space-x-3">
                <input
                  type="radio"
                  name="cognitiveLevel"
                  value={level.value}
                  checked={(editData.cognitiveLevel || user.cognitiveLevel) === level.value}
                  onChange={(e) => isEditing && updateEditData('cognitiveLevel', e.target.value)}
                  disabled={!isEditing}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 mt-1"
                />
                <div>
                  <span className="text-gray-900 dark:text-white font-medium">{level.label}</span>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{level.desc}</p>
                </div>
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Training Preferences */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Preferred Training Types</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {[
            'Breathing/Meditation', 'Games/Puzzles', 'Learning Videos',
            'Study Support Tools', 'Biofeedback Training', 'Cognitive Exercises'
          ].map((type) => (
            <button
              key={type}
              onClick={() => isEditing && toggleArrayValue('preferredTrainingTypes', type)}
              disabled={!isEditing}
              className={`p-3 text-sm rounded-lg border transition-colors ${
                (editData.preferredTrainingTypes || user.preferredTrainingTypes)?.includes(type)
                  ? 'bg-green-100 border-green-300 text-green-800 dark:bg-green-900/20 dark:border-green-600 dark:text-green-300'
                  : 'bg-gray-50 border-gray-300 text-gray-700 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300'
              } ${isEditing ? 'hover:bg-gray-100 dark:hover:bg-gray-600 cursor-pointer' : 'cursor-default'}`}
            >
              {type}
            </button>
          ))}
        </div>
      </div>
    </div>
  );

  const renderHealthTab = () => (
    <div className="space-y-8">
      {/* Health Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Sleep</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Average hours per night
              </label>
              {isEditing ? (
                <input
                  type="number"
                  value={editData.sleepHours || user.sleepHours || ''}
                  onChange={(e) => updateEditData('sleepHours', parseInt(e.target.value))}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  min="1"
                  max="12"
                />
              ) : (
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{user.sleepHours || 'Not set'}</p>
              )}
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Stress Level</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Current level (1-10)
              </label>
              {isEditing ? (
                <div className="space-y-2">
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={editData.stressLevel || user.stressLevel || 5}
                    onChange={(e) => updateEditData('stressLevel', parseInt(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-center text-2xl font-bold text-gray-900 dark:text-white">
                    {editData.stressLevel || user.stressLevel || 5}/10
                  </div>
                </div>
              ) : (
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{user.stressLevel || 'Not set'}/10</p>
              )}
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Exercise</h3>
          <div className="space-y-3">
            {[
              { value: 'never', label: 'Never' },
              { value: 'rarely', label: 'Rarely' },
              { value: 'weekly', label: 'Weekly' },
              { value: 'daily', label: 'Daily' }
            ].map((freq) => (
              <label key={freq.value} className="flex items-center space-x-3">
                <input
                  type="radio"
                  name="exerciseFrequency"
                  value={freq.value}
                  checked={(editData.exerciseFrequency || user.exerciseFrequency) === freq.value}
                  onChange={(e) => isEditing && updateEditData('exerciseFrequency', e.target.value)}
                  disabled={!isEditing}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                />
                <span className="text-gray-900 dark:text-white">{freq.label}</span>
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Medical Information */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Medical Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Medical Conditions
            </label>
            {isEditing ? (
              <textarea
                value={(editData.medicalConditions || user.medicalConditions || []).join(', ')}
                onChange={(e) => updateEditData('medicalConditions', e.target.value.split(', ').filter(Boolean))}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                rows={3}
                placeholder="Enter conditions separated by commas"
              />
            ) : (
              <p className="text-gray-900 dark:text-white">
                {user.medicalConditions?.length ? user.medicalConditions.join(', ') : 'None reported'}
              </p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Current Medications
            </label>
            {isEditing ? (
              <textarea
                value={(editData.medications || user.medications || []).join(', ')}
                onChange={(e) => updateEditData('medications', e.target.value.split(', ').filter(Boolean))}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                rows={3}
                placeholder="Enter medications separated by commas"
              />
            ) : (
              <p className="text-gray-900 dark:text-white">
                {user.medications?.length ? user.medications.join(', ') : 'None reported'}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Learning Style */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Learning Preferences</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
              Learning Style
            </label>
            <div className="space-y-3">
              {[
                { value: 'visual', label: 'Visual', desc: 'Learn through images and diagrams' },
                { value: 'auditory', label: 'Auditory', desc: 'Learn through sounds and speech' },
                { value: 'kinesthetic', label: 'Kinesthetic', desc: 'Learn through movement and touch' },
                { value: 'mixed', label: 'Mixed', desc: 'Combination of styles' }
              ].map((style) => (
                <label key={style.value} className="flex items-start space-x-3">
                  <input
                    type="radio"
                    name="learningStyle"
                    value={style.value}
                    checked={(editData.learningStyle || user.learningStyle) === style.value}
                    onChange={(e) => isEditing && updateEditData('learningStyle', e.target.value)}
                    disabled={!isEditing}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 mt-1"
                  />
                  <div>
                    <span className="text-gray-900 dark:text-white font-medium">{style.label}</span>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{style.desc}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
              Attention Span
            </label>
            <div className="space-y-3">
              {[
                { value: 'short', label: 'Short (5-15 min)', desc: 'Prefer brief sessions' },
                { value: 'medium', label: 'Medium (15-30 min)', desc: 'Standard session length' },
                { value: 'long', label: 'Long (30+ min)', desc: 'Extended focus periods' }
              ].map((span) => (
                <label key={span.value} className="flex items-start space-x-3">
                  <input
                    type="radio"
                    name="attentionSpan"
                    value={span.value}
                    checked={(editData.attentionSpan || user.attentionSpan) === span.value}
                    onChange={(e) => isEditing && updateEditData('attentionSpan', e.target.value)}
                    disabled={!isEditing}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 mt-1"
                  />
                  <div>
                    <span className="text-gray-900 dark:text-white font-medium">{span.label}</span>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{span.desc}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderDevicesTab = () => (
    <div className="space-y-8">
      {/* EEG Device Setup */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">EEG Device Configuration</h3>
        
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
              Do you have an EEG device?
            </label>
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => isEditing && updateEditData('hasEEGDevice', true)}
                disabled={!isEditing}
                className={`p-4 rounded-lg border transition-colors ${
                  (editData.hasEEGDevice ?? user.hasEEGDevice)
                    ? 'bg-blue-100 border-blue-300 text-blue-800 dark:bg-blue-900/20 dark:border-blue-600 dark:text-blue-300'
                    : 'bg-gray-50 border-gray-300 text-gray-700 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300'
                } ${isEditing ? 'hover:bg-gray-100 dark:hover:bg-gray-600 cursor-pointer' : 'cursor-default'}`}
              >
                Yes, I have one
              </button>
              <button
                onClick={() => isEditing && updateEditData('hasEEGDevice', false)}
                disabled={!isEditing}
                className={`p-4 rounded-lg border transition-colors ${
                  (editData.hasEEGDevice ?? user.hasEEGDevice) === false
                    ? 'bg-blue-100 border-blue-300 text-blue-800 dark:bg-blue-900/20 dark:border-blue-600 dark:text-blue-300'
                    : 'bg-gray-50 border-gray-300 text-gray-700 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300'
                } ${isEditing ? 'hover:bg-gray-100 dark:hover:bg-gray-600 cursor-pointer' : 'cursor-default'}`}
              >
                No, planning to get one
              </button>
            </div>
          </div>

          {(editData.hasEEGDevice ?? user.hasEEGDevice) && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Device Type
                </label>
                {isEditing ? (
                  <select
                    value={editData.eegDevice || user.eegDevice || ''}
                    onChange={(e) => updateEditData('eegDevice', e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  >
                    <option value="">Choose device</option>
                    <option value="Muse">Muse Headband</option>
                    <option value="Emotiv">Emotiv EPOC+</option>
                    <option value="NeuroSky">NeuroSky MindWave</option>
                    <option value="OpenBCI">OpenBCI</option>
                    <option value="Other">Other</option>
                  </select>
                ) : (
                  <p className="text-gray-900 dark:text-white">{user.eegDevice || 'Not specified'}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Device Model
                </label>
                {isEditing ? (
                  <input
                    type="text"
                    value={editData.deviceModel || user.deviceModel || ''}
                    onChange={(e) => updateEditData('deviceModel', e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    placeholder="e.g., Muse 2"
                  />
                ) : (
                  <p className="text-gray-900 dark:text-white">{user.deviceModel || 'Not specified'}</p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Training Environment */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Training Environment</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
              Preferred Session Duration
            </label>
            <div className="grid grid-cols-2 gap-3">
              {[
                { value: '5-10', label: '5-10 min' },
                { value: '10-20', label: '10-20 min' },
                { value: '20-30', label: '20-30 min' },
                { value: '30+', label: '30+ min' }
              ].map((duration) => (
                <button
                  key={duration.value}
                  onClick={() => isEditing && updateEditData('preferredSessionDuration', duration.value)}
                  disabled={!isEditing}
                  className={`p-3 text-sm rounded-lg border transition-colors ${
                    (editData.preferredSessionDuration || user.preferredSessionDuration) === duration.value
                      ? 'bg-blue-100 border-blue-300 text-blue-800 dark:bg-blue-900/20 dark:border-blue-600 dark:text-blue-300'
                      : 'bg-gray-50 border-gray-300 text-gray-700 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300'
                  } ${isEditing ? 'hover:bg-gray-100 dark:hover:bg-gray-600 cursor-pointer' : 'cursor-default'}`}
                >
                  {duration.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
              Sound Preference
            </label>
            <div className="space-y-2">
              {[
                { value: 'none', label: 'None' },
                { value: 'lo-fi', label: 'Lo-Fi Music' },
                { value: 'nature', label: 'Nature Sounds' },
                { value: 'white-noise', label: 'White Noise' },
                { value: 'binaural', label: 'Binaural Beats' }
              ].map((sound) => (
                <label key={sound.value} className="flex items-center space-x-3">
                  <input
                    type="radio"
                    name="soundPreference"
                    value={sound.value}
                    checked={(editData.soundPreference || user.soundPreference) === sound.value}
                    onChange={(e) => isEditing && updateEditData('soundPreference', e.target.value)}
                    disabled={!isEditing}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                  />
                  <span className="text-gray-900 dark:text-white">{sound.label}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-6">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
            Daily Training Window
          </label>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">From</label>
              {isEditing ? (
                <input
                  type="time"
                  value={editData.dailyTrainingWindow?.from || user.dailyTrainingWindow?.from || '09:00'}
                  onChange={(e) => updateEditData('dailyTrainingWindow', {
                    ...editData.dailyTrainingWindow,
                    from: e.target.value
                  })}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              ) : (
                <p className="text-gray-900 dark:text-white">{user.dailyTrainingWindow?.from || 'Not set'}</p>
              )}
            </div>
            <div>
              <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">To</label>
              {isEditing ? (
                <input
                  type="time"
                  value={editData.dailyTrainingWindow?.to || user.dailyTrainingWindow?.to || '17:00'}
                  onChange={(e) => updateEditData('dailyTrainingWindow', {
                    ...editData.dailyTrainingWindow,
                    to: e.target.value
                  })}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              ) : (
                <p className="text-gray-900 dark:text-white">{user.dailyTrainingWindow?.to || 'Not set'}</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderPreferencesTab = () => (
    <div className="space-y-8">
      {/* App Preferences */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">App Preferences</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Language
            </label>
            {isEditing ? (
              <select
                value={editData.language || user.language || ''}
                onChange={(e) => updateEditData('language', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                <option value="English">English</option>
                <option value="Spanish">Spanish</option>
                <option value="French">French</option>
                <option value="German">German</option>
                <option value="Hindi">Hindi</option>
                <option value="Tamil">Tamil</option>
              </select>
            ) : (
              <p className="text-gray-900 dark:text-white">{user.language}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Theme
            </label>
            {isEditing ? (
              <select
                value={editData.theme || user.theme || ''}
                onChange={(e) => updateEditData('theme', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                <option value="light">Light</option>
                <option value="dark">Dark</option>
                <option value="auto">Auto</option>
              </select>
            ) : (
              <p className="text-gray-900 dark:text-white capitalize">{user.theme}</p>
            )}
          </div>
        </div>
      </div>

      {/* Notification Settings */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Notifications</h3>
        
        <div className="space-y-4">
          {[
            { key: 'enableSmartTips', label: 'Smart Tips', description: 'Get personalized recommendations based on your brain state' },
            { key: 'allowNotifications', label: 'Training Reminders', description: 'Receive notifications for scheduled training sessions' },
            { key: 'personalizedContent', label: 'Personalized Content', description: 'Show content adapted to your brainwave patterns' },
            { key: 'marketingConsent', label: 'Marketing Updates', description: 'Receive updates about new features and research findings' }
          ].map((setting) => (
            <div key={setting.key} className="flex items-start justify-between">
              <div className="flex-1">
                <h4 className="text-sm font-medium text-gray-900 dark:text-white">{setting.label}</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">{setting.description}</p>
              </div>
              <button
                onClick={() => isEditing && updateEditData(setting.key, !(editData[setting.key as keyof UserProfile] ?? user[setting.key as keyof UserProfile]))}
                disabled={!isEditing}
                className={`w-12 h-6 rounded-full transition-colors ${
                  (editData[setting.key as keyof UserProfile] ?? user[setting.key as keyof UserProfile])
                    ? 'bg-blue-600'
                    : 'bg-gray-300 dark:bg-gray-600'
                } ${isEditing ? 'cursor-pointer' : 'cursor-default'}`}
              >
                <div
                  className={`w-5 h-5 bg-white rounded-full shadow transition-transform ${
                    (editData[setting.key as keyof UserProfile] ?? user[setting.key as keyof UserProfile])
                      ? 'translate-x-6'
                      : 'translate-x-0.5'
                  }`}
                />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderPrivacyTab = () => (
    <div className="space-y-8">
      {/* Privacy Settings */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Privacy & Security</h3>
        
        <div className="space-y-6">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <h4 className="text-sm font-medium text-gray-900 dark:text-white">Data Sharing</h4>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                Allow anonymous data contribution to neuroscience research
              </p>
            </div>
            <button
              onClick={() => isEditing && updateEditData('dataSharing', !editData.dataSharing)}
              disabled={!isEditing}
              className={`w-12 h-6 rounded-full transition-colors ${
                (editData.dataSharing ?? user.dataSharing)
                  ? 'bg-blue-600'
                  : 'bg-gray-300 dark:bg-gray-600'
              } ${isEditing ? 'cursor-pointer' : 'cursor-default'}`}
            >
              <div
                className={`w-5 h-5 bg-white rounded-full shadow transition-transform ${
                  (editData.dataSharing ?? user.dataSharing)
                    ? 'translate-x-6'
                    : 'translate-x-0.5'
                }`}
              />
            </button>
          </div>

          <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
            <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-4">Account Actions</h4>
            <div className="space-y-3">
              <button className="flex items-center space-x-3 text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors">
                <Download className="h-4 w-4" />
                <span>Download My Data</span>
              </button>
              <button className="flex items-center space-x-3 text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors">
                <Upload className="h-4 w-4" />
                <span>Import Data</span>
              </button>
              <button 
                onClick={() => setShowDeleteConfirm(true)}
                className="flex items-center space-x-3 text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 transition-colors"
              >
                <Trash2 className="h-4 w-4" />
                <span>Delete Account</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Consent Status */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Consent Status</h3>
        
        <div className="space-y-4">
          {[
            { key: 'consentGiven', label: 'Terms of Service', status: user.consentGiven },
            { key: 'privacyPolicyAccepted', label: 'Privacy Policy', status: user.privacyPolicyAccepted },
            { key: 'marketingConsent', label: 'Marketing Communications', status: user.marketingConsent }
          ].map((consent) => (
            <div key={consent.key} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <span className="text-sm font-medium text-gray-900 dark:text-white">{consent.label}</span>
              <div className="flex items-center space-x-2">
                {consent.status ? (
                  <CheckCircle className="h-5 w-5 text-green-500" />
                ) : (
                  <AlertCircle className="h-5 w-5 text-yellow-500" />
                )}
                <span className={`text-sm ${consent.status ? 'text-green-600' : 'text-yellow-600'}`}>
                  {consent.status ? 'Accepted' : 'Pending'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderAchievementsTab = () => (
    <div className="space-y-8">
      {/* Achievement Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {[
          { label: 'Total Points', value: '2,847', icon: Award, color: 'yellow' },
          { label: 'Badges Earned', value: '12', icon: CheckCircle, color: 'green' },
          { label: 'Streak Record', value: '28 days', icon: TrendingUp, color: 'blue' },
          { label: 'Level', value: 'Expert', icon: Brain, color: 'purple' }
        ].map((stat, index) => {
          const Icon = stat.icon;
          return (
            <div key={index} className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{stat.label}</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">{stat.value}</p>
                </div>
                <div className={`p-3 rounded-lg bg-${stat.color}-100 dark:bg-${stat.color}-900/20`}>
                  <Icon className={`h-6 w-6 text-${stat.color}-600 dark:text-${stat.color}-400`} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Recent Achievements */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Recent Achievements</h3>
        
        <div className="space-y-4">
          {[
            { 
              title: 'Focus Master', 
              description: 'Completed 50 focus training sessions', 
              date: '2 days ago',
              icon: Target,
              color: 'blue'
            },
            { 
              title: 'Meditation Guru', 
              description: 'Achieved 100 hours of meditation', 
              date: '1 week ago',
              icon: Brain,
              color: 'green'
            },
            { 
              title: 'Consistency Champion', 
              description: 'Maintained a 14-day training streak', 
              date: '2 weeks ago',
              icon: TrendingUp,
              color: 'purple'
            }
          ].map((achievement, index) => {
            const Icon = achievement.icon;
            return (
              <div key={index} className="flex items-center space-x-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className={`p-3 rounded-full bg-${achievement.color}-100 dark:bg-${achievement.color}-900/20`}>
                  <Icon className={`h-6 w-6 text-${achievement.color}-600 dark:text-${achievement.color}-400`} />
                </div>
                <div className="flex-1">
                  <h4 className="font-medium text-gray-900 dark:text-white">{achievement.title}</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{achievement.description}</p>
                </div>
                <span className="text-sm text-gray-500">{achievement.date}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview': return renderOverviewTab();
      case 'goals': return renderGoalsTab();
      case 'health': return renderHealthTab();
      case 'devices': return renderDevicesTab();
      case 'preferences': return renderPreferencesTab();
      case 'privacy': return renderPrivacyTab();
      case 'achievements': return renderAchievementsTab();
      default: return renderOverviewTab();
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Profile Settings
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Manage your personal information and preferences
          </p>
        </div>
        
        <div className="flex items-center space-x-3">
          {isEditing ? (
            <>
              <button
                onClick={handleCancel}
                className="flex items-center space-x-2 px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
              >
                <X className="h-4 w-4" />
                <span>Cancel</span>
              </button>
              <motion.button
                onClick={handleSave}
                disabled={isLoading}
                className="flex items-center space-x-2 bg-blue-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors disabled:opacity-50"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {isLoading ? (
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  <Save className="h-4 w-4" />
                )}
                <span>Save Changes</span>
              </motion.button>
            </>
          ) : (
            <motion.button
              onClick={() => setIsEditing(true)}
              className="flex items-center space-x-2 bg-blue-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Edit3 className="h-4 w-4" />
              <span>Edit Profile</span>
            </motion.button>
          )}
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="flex space-x-8 overflow-x-auto">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
        >
          {renderTabContent()}
        </motion.div>
      </AnimatePresence>

      {/* Delete Account Confirmation Modal */}
      <AnimatePresence>
        {showDeleteConfirm && (
          <motion.div
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="bg-white dark:bg-gray-800 rounded-xl p-6 max-w-md w-full"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
            >
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 bg-red-100 dark:bg-red-900/20 rounded-full">
                  <AlertCircle className="h-6 w-6 text-red-600 dark:text-red-400" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Delete Account
                </h3>
              </div>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                Are you sure you want to delete your account? This action cannot be undone and all your data will be permanently removed.
              </p>
              <div className="flex space-x-3">
                <button
                  onClick={() => setShowDeleteConfirm(false)}
                  className="flex-1 px-4 py-2 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                >
                  Cancel
                </button>
                <button className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors">
                  Delete Account
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Profile;