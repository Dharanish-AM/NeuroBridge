import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Activity, 
  BookOpen, 
  User,
  Menu
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useBrainData } from '@/context/BrainDataContext';
import { useState } from 'react';

const Sidebar: React.FC = () => {
  const { mentalState } = useBrainData();
  const [collapsed, setCollapsed] = useState(false);

  const navItems = [
    { name: 'Dashboard', path: '/dashboard', icon: <LayoutDashboard className="h-5 w-5" /> },
    { name: 'EEG Session', path: '/eeg-session', icon: <Activity className="h-5 w-5" /> },
    { name: 'Lessons', path: '/lessons', icon: <BookOpen className="h-5 w-5" /> },
    { name: 'Profile', path: '/profile', icon: <User className="h-5 w-5" /> }
  ];

  // Get the color based on mental state
  const getMentalStateColor = (state: string) => {
    switch (state) {
      case 'Focused': return 'bg-green-500';
      case 'Relaxed': return 'bg-blue-400';
      case 'Drowsy': return 'bg-orange-400';
      default: return 'bg-gray-400';
    }
  };

  return (
    <div 
      className={cn(
        "bg-white border-r border-gray-200 flex flex-col transition-all duration-300 ease-in-out",
        collapsed ? "w-20" : "w-64"
      )}
    >
      {/* Logo and App Name */}
      <div className="p-4 border-b flex items-center gap-3">
        <div className="bg-indigo-600 text-white rounded-lg w-10 h-10 flex items-center justify-center font-bold text-xl">
          N+
        </div>
        {!collapsed && <span className="font-bold text-xl">NeuroBridge+</span>}
      </div>

      {/* Current Mental State Indicator */}
      <div className={cn(
        "mx-4 my-3 rounded-lg p-3",
        collapsed ? "items-center justify-center" : "",
        getMentalStateColor(mentalState)
      )}>
        <div className="flex items-center gap-2 text-white">
          <div className="h-3 w-3 rounded-full bg-white animate-pulse"></div>
          {!collapsed && (
            <>
              <span className="font-medium">Current State:</span>
              <span className="font-semibold">{mentalState}</span>
            </>
          )}
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 py-4">
        <ul className="space-y-1">
          {navItems.map((item) => (
            <li key={item.path} className="px-3">
              <NavLink
                to={item.path}
                className={({ isActive }) => cn(
                  "flex items-center gap-3 px-3 py-2 rounded-lg transition-all",
                  isActive 
                    ? "bg-indigo-50 text-indigo-700 font-medium" 
                    : "text-gray-600 hover:bg-gray-100",
                  collapsed && "justify-center"
                )}
              >
                {item.icon}
                {!collapsed && <span>{item.name}</span>}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* Collapse button */}
      <div className="p-4 border-t">
        <Button 
          variant="ghost" 
          size="sm" 
          className="w-full flex items-center justify-center"
          onClick={() => setCollapsed(prev => !prev)}
        >
          <Menu className="h-5 w-5" />
          {!collapsed && <span className="ml-2">Collapse</span>}
        </Button>
      </div>
    </div>
  );
};

export default Sidebar;