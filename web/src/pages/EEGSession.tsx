import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useBrainData, ChannelData } from '@/context/BrainDataContext';
import EEGChannelChart from '@/components/eeg/EEGChannelChart';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Play, Pause, Save } from 'lucide-react';

const EEGSession: React.FC = () => {
  const { channels, mentalState, brainwaves } = useBrainData();
  const [isRecording, setIsRecording] = useState<boolean>(true);
  const [showFiltered, setShowFiltered] = useState<boolean>(true);
  const [selectedTab, setSelectedTab] = useState<string>("all");

  // Group channels for display
  const groupedChannels = {
    all: channels,
    frontal: channels.filter(c => c.name.startsWith('F') || c.name.startsWith('Fp')),
    central: channels.filter(c => c.name.startsWith('C')),
    temporal: channels.filter(c => c.name.startsWith('T')),
    occipital: channels.filter(c => c.name.startsWith('O')),
  };

  // Determine mental state badge color
  const getMentalStateBadgeVariant = (state: string) => {
    switch (state) {
      case 'Focused': return 'default'; // Blue in shadcn
      case 'Relaxed': return 'secondary'; // Gray
      case 'Drowsy': return 'destructive'; // Red
      default: return 'outline';
    }
  };

  return (
    <div className="container mx-auto space-y-8">
      <div className="flex justify-between items-center">
        <h2 className="text-3xl font-bold tracking-tight">Live EEG Session</h2>
        <div className="flex items-center gap-4">
          <Badge variant={getMentalStateBadgeVariant(mentalState)}>
            Current State: {mentalState}
          </Badge>
          <div className="flex items-center space-x-2">
            <Switch
              id="filter-mode"
              checked={showFiltered}
              onCheckedChange={setShowFiltered}
            />
            <Label htmlFor="filter-mode">Show Filtered Data</Label>
          </div>
        </div>
      </div>

      {/* Recording controls */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex justify-between items-center">
            <div>
              <CardTitle>Session Controls</CardTitle>
              <CardDescription>Manage your EEG recording session</CardDescription>
            </div>
            <div className="flex gap-2">
              <Button 
                variant={isRecording ? "destructive" : "default"}
                size="sm"
                onClick={() => setIsRecording(!isRecording)}
              >
                {isRecording ? <Pause className="mr-1 h-4 w-4" /> : <Play className="mr-1 h-4 w-4" />}
                {isRecording ? "Pause" : "Start"} Recording
              </Button>
              <Button variant="outline" size="sm">
                <Save className="mr-1 h-4 w-4" />
                Save Session
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-5 gap-4">
            {brainwaves.map(wave => (
              <Card key={wave.name} className="p-3 text-center">
                <h3 className="font-medium">{wave.name}</h3>
                <p className="text-2xl font-bold" style={{ color: wave.color }}>
                  {wave.value}%
                </p>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* EEG Channel Tabs */}
      <Tabs defaultValue="all" value={selectedTab} onValueChange={setSelectedTab} className="w-full">
        <TabsList className="grid grid-cols-5">
          <TabsTrigger value="all">All Channels</TabsTrigger>
          <TabsTrigger value="frontal">Frontal</TabsTrigger>
          <TabsTrigger value="central">Central</TabsTrigger>
          <TabsTrigger value="temporal">Temporal</TabsTrigger>
          <TabsTrigger value="occipital">Occipital</TabsTrigger>
        </TabsList>
        
        {Object.entries(groupedChannels).map(([key, channelGroup]) => (
          <TabsContent key={key} value={key} className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle>{key === 'all' ? 'All EEG Channels' : `${key.charAt(0).toUpperCase() + key.slice(1)} Channels`}</CardTitle>
                <CardDescription>
                  {showFiltered ? 'Filtered' : 'Raw'} waveform data display
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {channelGroup.map((channel: ChannelData) => (
                    <div key={channel.name} className="h-40">
                      <EEGChannelChart 
                        channel={channel} 
                        filtered={showFiltered} 
                      />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        ))}
      </Tabs>

      {/* Session Info */}
      <Card>
        <CardHeader>
          <CardTitle>Session Information</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-1">
              <dt className="text-sm font-medium text-muted-foreground">Duration</dt>
              <dd className="text-xl font-bold">00:15:23</dd>
            </div>
            <div className="space-y-1">
              <dt className="text-sm font-medium text-muted-foreground">Sampling Rate</dt>
              <dd className="text-xl font-bold">256 Hz</dd>
            </div>
            <div className="space-y-1">
              <dt className="text-sm font-medium text-muted-foreground">Connection Quality</dt>
              <dd className="text-xl font-bold">Excellent</dd>
            </div>
          </dl>
        </CardContent>
      </Card>
    </div>
  );
};

export default EEGSession;