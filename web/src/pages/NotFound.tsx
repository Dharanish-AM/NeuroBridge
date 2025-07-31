import React from 'react';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';

const NotFound: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 px-4">
      <div className="text-center space-y-6">
        <h1 className="text-9xl font-bold text-indigo-600">404</h1>
        <h2 className="text-3xl font-semibold text-gray-900">Page Not Found</h2>
        <p className="text-gray-600 max-w-md mx-auto">
          The page you're looking for doesn't exist or has been moved to another URL.
        </p>
        <div className="flex justify-center gap-4">
          <Button asChild variant="default">
            <Link to="/">Back to Dashboard</Link>
          </Button>
          <Button asChild variant="outline">
            <Link to="/eeg-session">Start a Session</Link>
          </Button>
        </div>
      </div>
    </div>
  );
};

export default NotFound;