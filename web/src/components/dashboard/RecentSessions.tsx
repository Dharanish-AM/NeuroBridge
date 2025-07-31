import React from 'react';
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

// Mock data for recent sessions
const mockSessions = [
  {
    id: 'S-001',
    date: '2025-07-30',
    duration: '45 min',
    avgFocus: 78,
    mentalState: 'Focused',
    module: 'Advanced Mathematics'
  },
  {
    id: 'S-002',
    date: '2025-07-29',
    duration: '30 min',
    avgFocus: 65,
    mentalState: 'Relaxed',
    module: 'History of Science'
  },
  {
    id: 'S-003',
    date: '2025-07-28',
    duration: '60 min',
    avgFocus: 45,
    mentalState: 'Drowsy',
    module: 'Literature Analysis'
  },
  {
    id: 'S-004',
    date: '2025-07-27',
    duration: '25 min',
    avgFocus: 82,
    mentalState: 'Focused',
    module: 'Physics Fundamentals'
  },
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

const RecentSessions: React.FC = () => {
  return (
    <Table>
      <TableCaption>Recent EEG monitoring sessions</TableCaption>
      <TableHeader>
        <TableRow>
          <TableHead>Session ID</TableHead>
          <TableHead>Date</TableHead>
          <TableHead>Duration</TableHead>
          <TableHead>Avg. Focus</TableHead>
          <TableHead>Mental State</TableHead>
          <TableHead>Learning Module</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {mockSessions.map((session) => (
          <TableRow key={session.id}>
            <TableCell className="font-medium">{session.id}</TableCell>
            <TableCell>{session.date}</TableCell>
            <TableCell>{session.duration}</TableCell>
            <TableCell>{session.avgFocus}%</TableCell>
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
  );
};

export default RecentSessions;