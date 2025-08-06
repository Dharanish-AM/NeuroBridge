import React from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { AreaChart, BarChart, Calendar, User, Settings, Book } from 'lucide-react';
import { useBrainData } from '@/context/BrainDataContext';
import MentalStateHistory from '@/components/profile/MentalStateHistory';
import SessionsHistory from '@/components/profile/SessionsHistory';

const Profile: React.FC = () => {
  const { mentalState } = useBrainData();
  
  return (
    <div className="container mx-auto space-y-8">
      <div className="flex justify-between items-center">
        <h2 className="text-3xl font-bold tracking-tight">Profile</h2>
      </div>

      {/* User Info Card */}
      <div className="grid gap-6 md:grid-cols-3">
        {/* Profile Summary */}
        <Card className="md:col-span-1">
          <CardHeader>
            <CardTitle>Personal Information</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col items-center text-center">
            <Avatar className="h-32 w-32 mb-4">
              <AvatarImage src="/images/UserAvatar.png" alt="User" />
              <AvatarFallback className="text-3xl">JD</AvatarFallback>
            </Avatar>
            <h3 className="text-2xl font-bold mb-1">Dharanish A M</h3>
            <p className="text-muted-foreground mb-4">College Student</p>
            
            <div className="grid grid-cols-3 gap-4 w-full mb-4">
              <div className="text-center">
                <p className="text-xl font-bold">127</p>
                <p className="text-xs text-muted-foreground">Sessions</p>
              </div>
              <div className="text-center">
                <p className="text-xl font-bold">48</p>
                <p className="text-xs text-muted-foreground">Lessons</p>
              </div>
              <div className="text-center">
                <p className="text-xl font-bold">75%</p>
                <p className="text-xs text-muted-foreground">Avg Focus</p>
              </div>
            </div>
            
            <Button variant="outline" size="sm" className="w-full">Edit Profile</Button>
          </CardContent>
        </Card>

        {/* Profile Details */}
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Account Details</CardTitle>
            <CardDescription>Update your personal information</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="firstName">First Name</Label>
                <Input id="firstName" defaultValue="Dharanish" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="lastName">Last Name</Label>
                <Input id="lastName" defaultValue="A M" />
              </div>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input id="email" type="email" defaultValue="dharanish816@gmail.com" />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="parentPhone">Parent Phone Number</Label>
              <Input id="parentPhone" type="tel" defaultValue="+918668030261" placeholder="+91XXXXXXXXXX" />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="education">Education Level</Label>
              <Input id="education" defaultValue="Bachelor's - Computer Science" />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="bio">Bio</Label>
              <Textarea id="bio" rows={3} defaultValue="Computer Science student with interest in artificial intelligence and cognitive learning. Looking to enhance my study habits and cognitive performance." />
            </div>
          </CardContent>
          <CardFooter>
            <Button>Save Changes</Button>
          </CardFooter>
        </Card>
      </div>

      {/* Tabs for Additional Information */}
      <Tabs defaultValue="mental-history">
        <TabsList className="grid grid-cols-4">
          <TabsTrigger value="mental-history">
            <AreaChart className="h-4 w-4 mr-2" />
            Mental State History
          </TabsTrigger>
          <TabsTrigger value="sessions">
            <Calendar className="h-4 w-4 mr-2" />
            Past Sessions
          </TabsTrigger>
          <TabsTrigger value="achievements">
            <Book className="h-4 w-4 mr-2" />
            Learning Progress
          </TabsTrigger>
          <TabsTrigger value="settings">
            <Settings className="h-4 w-4 mr-2" />
            Preferences
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="mental-history" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Mental State History</CardTitle>
              <CardDescription>
                Your cognitive performance over time
              </CardDescription>
            </CardHeader>
            <CardContent>
              <MentalStateHistory />
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="sessions" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Past Learning Sessions</CardTitle>
              <CardDescription>
                View your previous EEG monitoring sessions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <SessionsHistory />
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="achievements" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Learning Progress</CardTitle>
              <CardDescription>
                Track your course completion and achievements
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div>
                  <h3 className="font-medium mb-2">Course Completion</h3>
                  <div className="w-full h-4 rounded-full bg-gray-100 overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full"
                      style={{ width: '65%' }}
                    ></div>
                  </div>
                  <div className="flex justify-between mt-1 text-sm text-muted-foreground">
                    <span>0%</span>
                    <span>Overall Progress: 65%</span>
                    <span>100%</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {/* Mock achievements */}
                  {['Focused Learner', 'Quick Adapter', '7-Day Streak', 'Brain Master'].map((achievement, i) => (
                    <Card key={i} className="text-center p-4">
                      <div className="w-12 h-12 mx-auto mb-2 rounded-full bg-indigo-100 flex items-center justify-center">
                        <BarChart className="h-6 w-6 text-indigo-600" />
                      </div>
                      <h4 className="font-medium">{achievement}</h4>
                      <p className="text-xs text-muted-foreground">Unlocked</p>
                    </Card>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="settings" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Preferences</CardTitle>
              <CardDescription>
                Customize your learning experience
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="space-y-4">
                  <h3 className="font-medium">Notification Preferences</h3>
                  <div className="grid grid-cols-2 gap-4">
                    {/* Notification options */}
                    {[
                      'Session reminders', 
                      'Focus alerts', 
                      'Lesson recommendations', 
                      'Progress reports'
                    ].map((notification, i) => (
                      <div key={i} className="flex items-center space-x-2">
                        <input type="checkbox" id={`notification-${i}`} defaultChecked className="h-4 w-4" />
                        <Label htmlFor={`notification-${i}`}>{notification}</Label>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div className="space-y-4">
                  <h3 className="font-medium">Display Settings</h3>
                  <div className="space-y-2">
                    <Label htmlFor="theme">Theme</Label>
                    <select id="theme" className="w-full p-2 border rounded">
                      <option>Light</option>
                      <option>Dark</option>
                      <option>System default</option>
                    </select>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <h3 className="font-medium">Device Connection</h3>
                  <Button variant="outline">Connect EEG Device</Button>
                  <p className="text-xs text-muted-foreground">Last connected: Today, 10:23 AM</p>
                </div>
              </div>
            </CardContent>
            <CardFooter>
              <Button>Save Preferences</Button>
            </CardFooter>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Profile;