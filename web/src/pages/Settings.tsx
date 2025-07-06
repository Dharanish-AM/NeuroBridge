import React from 'react';
import { motion } from 'framer-motion';
import { Settings as SettingsIcon, Brain, Bell, Palette, Shield, Wifi, Download, Trash2 } from 'lucide-react';
import { useTheme } from '../hooks/useTheme';

const Settings: React.FC = () => {
  const { theme, toggleTheme } = useTheme();

  const settingsSections = [
    {
      id: 'device',
      title: 'Device Settings',
      icon: Brain,
      items: [
        {
          label: 'EEG Device',
          description: 'Configure your EEG headset connection',
          type: 'select',
          options: ['NeuroSky MindWave', 'Emotiv EPOC+', 'Muse Headband', 'OpenBCI'],
          value: 'Muse Headband'
        },
        {
          label: 'Sampling Rate',
          description: 'Data collection frequency',
          type: 'select',
          options: ['256 Hz', '512 Hz', '1024 Hz'],
          value: '256 Hz'
        },
        {
          label: 'Signal Quality Threshold',
          description: 'Minimum signal quality for training',
          type: 'range',
          min: 50,
          max: 100,
          value: 80
        }
      ]
    },
    {
      id: 'notifications',
      title: 'Notifications',
      icon: Bell,
      items: [
        {
          label: 'Training Reminders',
          description: 'Daily training session notifications',
          type: 'toggle',
          value: true
        },
        {
          label: 'Focus Alerts',
          description: 'Alert when attention drops during sessions',
          type: 'toggle',
          value: true
        },
        {
          label: 'Achievement Notifications',
          description: 'Celebrate your progress milestones',
          type: 'toggle',
          value: true
        },
        {
          label: 'Weekly Reports',
          description: 'Email summary of your weekly progress',
          type: 'toggle',
          value: false
        }
      ]
    },
    {
      id: 'appearance',
      title: 'Appearance',
      icon: Palette,
      items: [
        {
          label: 'Theme',
          description: 'Choose your preferred theme',
          type: 'select',
          options: ['Light', 'Dark', 'Auto'],
          value: theme === 'light' ? 'Light' : 'Dark'
        },
        {
          label: 'Chart Animation',
          description: 'Enable smooth chart transitions',
          type: 'toggle',
          value: true
        },
        {
          label: 'Reduced Motion',
          description: 'Minimize animations for accessibility',
          type: 'toggle',
          value: false
        }
      ]
    },
    {
      id: 'privacy',
      title: 'Privacy & Data',
      icon: Shield,
      items: [
        {
          label: 'Data Collection',
          description: 'Allow anonymous usage analytics',
          type: 'toggle',
          value: true
        },
        {
          label: 'Local Storage Only',
          description: 'Keep all data on your device',
          type: 'toggle',
          value: false
        },
        {
          label: 'Auto-delete Old Sessions',
          description: 'Remove session data older than 90 days',
          type: 'toggle',
          value: true
        }
      ]
    }
  ];

  const renderSettingInput = (item: any) => {
    switch (item.type) {
      case 'toggle':
        return (
          <button
            className={`w-12 h-6 rounded-full transition-colors ${
              item.value ? 'bg-blue-600' : 'bg-gray-300 dark:bg-gray-600'
            }`}
          >
            <div
              className={`w-5 h-5 bg-white rounded-full shadow transition-transform ${
                item.value ? 'translate-x-6' : 'translate-x-0.5'
              }`}
            />
          </button>
        );
      case 'select':
        return (
          <select className="bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1 text-sm">
            {item.options.map((option: string) => (
              <option key={option} value={option} selected={option === item.value}>
                {option}
              </option>
            ))}
          </select>
        );
      case 'range':
        return (
          <div className="flex items-center space-x-3">
            <input
              type="range"
              min={item.min}
              max={item.max}
              value={item.value}
              className="flex-1"
            />
            <span className="text-sm font-medium text-gray-900 dark:text-white w-10">
              {item.value}%
            </span>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Settings
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Customize your NeuroBridge experience and manage your preferences
        </p>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[
          { icon: Wifi, label: 'Test Connection', action: 'Test your EEG device connection' },
          { icon: Download, label: 'Export Data', action: 'Download your training data' },
          { icon: Palette, label: 'Toggle Theme', action: 'Switch between light and dark mode', onClick: toggleTheme },
          { icon: Trash2, label: 'Clear Cache', action: 'Remove temporary files and reset' }
        ].map((action, index) => (
          <motion.button
            key={action.label}
            className="p-4 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 text-left hover:shadow-md transition-shadow"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
            whileHover={{ scale: 1.02 }}
            onClick={action.onClick}
          >
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
                <action.icon className="h-4 w-4 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <div className="font-medium text-gray-900 dark:text-white text-sm">
                  {action.label}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  {action.action}
                </div>
              </div>
            </div>
          </motion.button>
        ))}
      </div>

      {/* Settings Sections */}
      <div className="space-y-8">
        {settingsSections.map((section, sectionIndex) => {
          const Icon = section.icon;
          return (
            <motion.div
              key={section.id}
              className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 + sectionIndex * 0.1 }}
            >
              <div className="flex items-center space-x-3 mb-6">
                <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
                  <Icon className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  {section.title}
                </h3>
              </div>
              
              <div className="space-y-6">
                {section.items.map((item, itemIndex) => (
                  <motion.div
                    key={item.label}
                    className="flex items-center justify-between py-3 border-b border-gray-100 dark:border-gray-700 last:border-b-0"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: 0.5 + sectionIndex * 0.1 + itemIndex * 0.05 }}
                  >
                    <div className="flex-1">
                      <h4 className="font-medium text-gray-900 dark:text-white">
                        {item.label}
                      </h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        {item.description}
                      </p>
                    </div>
                    <div className="ml-4">
                      {renderSettingInput(item)}
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Data Management */}
      <motion.div
        className="bg-gradient-to-r from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20 rounded-xl p-6 border border-red-200 dark:border-red-800"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.8 }}
      >
        <div className="flex items-start space-x-4">
          <div className="p-2 bg-red-100 dark:bg-red-900/20 rounded-lg">
            <Trash2 className="h-5 w-5 text-red-600 dark:text-red-400" />
          </div>
          <div className="flex-1">
            <h4 className="font-semibold text-red-900 dark:text-red-100 mb-2">
              Data Management
            </h4>
            <p className="text-red-700 dark:text-red-300 text-sm mb-4">
              Manage your stored data and account information. These actions cannot be undone.
            </p>
            <div className="flex space-x-3">
              <button className="bg-red-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-red-700 transition-colors">
                Clear All Data
              </button>
              <button className="border border-red-300 dark:border-red-700 text-red-700 dark:text-red-300 px-4 py-2 rounded-lg text-sm font-medium hover:bg-red-50 dark:hover:bg-red-900/10 transition-colors">
                Export Before Clearing
              </button>
            </div>
          </div>
        </div>
      </motion.div>

      {/* App Info */}
      <motion.div
        className="text-center py-8 border-t border-gray-200 dark:border-gray-700"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 1 }}
      >
        <div className="text-gray-500 dark:text-gray-400 text-sm">
          <p className="mb-2">NeuroBridge v1.0.0</p>
          <p>Built with ❤️ for better mental performance</p>
        </div>
      </motion.div>
    </div>
  );
};

export default Settings;