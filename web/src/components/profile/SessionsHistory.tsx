import React, { useState } from 'react';
import { 
  Table, 
  TableBody, 
  TableCaption, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Calendar as CalendarIcon, Search, Download } from 'lucide-react';

// Mock data for sessions history
const mockSessions = [
  {
    id: 'S-001',
    date: '2025-07-30',
    time: '10:30 AM',
    duration: '45 min',
    avgFocus: 78,
    mentalState: 'Focused',
    module: 'Advanced Mathematics'
  },
  {
    id: 'S-002',
    date: '2025-07-29',
    time: '2:15 PM',
    duration: '30 min',
    avgFocus: 65,
    mentalState: 'Relaxed',
    module: 'History of Science'
  },
  {
    id: 'S-003',
    date: '2025-07-28',
    time: '9:00 AM',
    duration: '60 min',
    avgFocus: 45,
    mentalState: 'Drowsy',
    module: 'Literature Analysis'
  },
  {
    id: 'S-004',
    date: '2025-07-27',
    time: '4:45 PM',
    duration: '25 min',
    avgFocus: 82,
    mentalState: 'Focused',
    module: 'Physics Fundamentals'
  },
  {
    id: 'S-005',
    date: '2025-07-26',
    time: '11:20 AM',
    duration: '40 min',
    avgFocus: 70,
    mentalState: 'Focused',
    module: 'Computer Science'
  },
  {
    id: 'S-006',
    date: '2025-07-25',
    time: '3:30 PM',
    duration: '50 min',
    avgFocus: 58,
    mentalState: 'Relaxed',
    module: 'Foreign Language'
  },
  {
    id: 'S-007',
    date: '2025-07-24',
    time: '10:00 AM',
    duration: '35 min',
    avgFocus: 63,
    mentalState: 'Relaxed',
    module: 'Chemistry Basics'
  }
];

// Helper function to get badge variant based on mental state
const getMentalStateBadgeVariant = (state: string) => {
  switch (state) {
    case 'Focused': return 'default'; // Blue in shadcn
    case 'Relaxed': return 'secondary'; // Gray
    case 'Drowsy': return 'destructive'; // Red
    default: return 'outline';
  }
};

const SessionsHistory: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  
  // Filter sessions based on search term
  const filteredSessions = mockSessions.filter(session => 
    session.module.toLowerCase().includes(searchTerm.toLowerCase()) ||
    session.date.includes(searchTerm) ||
    session.mentalState.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-4">
      {/* Search and filter */}
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input 
            placeholder="Search by module, date, or state..." 
            className="pl-8"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <Button variant="outline" size="icon">
          <CalendarIcon className="h-4 w-4" />
        </Button>
        <Button variant="outline" size="icon">
          <Download className="h-4 w-4" />
        </Button>
      </div>
      
      {/* Sessions table */}
      <Table>
        <TableCaption>A history of your EEG monitoring sessions</TableCaption>
        <TableHeader>
          <TableRow>
            <TableHead>Session ID</TableHead>
            <TableHead>Date & Time</TableHead>
            <TableHead>Duration</TableHead>
            <TableHead>Focus Level</TableHead>
            <TableHead>Mental State</TableHead>
            <TableHead>Learning Module</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {filteredSessions.map((session) => (
            <TableRow key={session.id} className="cursor-pointer hover:bg-gray-50">
              <TableCell className="font-medium">{session.id}</TableCell>
              <TableCell>{session.date}<br/><span className="text-xs text-muted-foreground">{session.time}</span></TableCell>
              <TableCell>{session.duration}</TableCell>
              <TableCell>
                <div className="flex items-center gap-2">
                  <div className="w-10 h-2 rounded-full bg-gray-200">
                    <div 
                      className={`h-full rounded-full ${
                        session.avgFocus > 70 ? 'bg-green-500' : 
                        session.avgFocus > 50 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${session.avgFocus}%` }}
                    ></div>
                  </div>
                  <span>{session.avgFocus}%</span>
                </div>
              </TableCell>
              <TableCell>
                <Badge variant={getMentalStateBadgeVariant(session.mentalState)}>
                  {session.mentalState}
                </Badge>
              </TableCell>
              <TableCell>{session.module}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
};

export default SessionsHistory;