import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Clock, BookOpen, CheckCircle, Star, Award, Brain, Search, Filter, PlayCircle, ArrowRight } from 'lucide-react';
import { useBrainData } from '@/context/BrainDataContext';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Progress } from '@/components/ui/progress';
import { Avatar, AvatarImage, AvatarFallback } from '@/components/ui/avatar';

interface Lesson {
  id: string;
  title: string;
  description: string;
  duration: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  status: 'New' | 'In Progress' | 'Completed';
  completion: number;
  recommended: boolean;
  mentalState: string;
  tags: string[];
  brainFocus: string[];
  content?: string[];
  objectives?: string[];
  prerequisites?: string[];
  relatedLessons?: string[];
}

const LessonRecommendations: React.FC = () => {
  const { mentalState } = useBrainData();
  const [filter, setFilter] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedLesson, setSelectedLesson] = useState<Lesson | null>(null);
  const [activeLesson, setActiveLesson] = useState<Lesson | null>(null);
  const [isLessonOpen, setIsLessonOpen] = useState<boolean>(false);
  const [lessonProgress, setLessonProgress] = useState<number>(0);
  const [currentSection, setCurrentSection] = useState<number>(0);
  
  // Mock lesson data
  const [lessons, setLessons] = useState<Lesson[]>([
    {
      id: 'L1',
      title: 'Deep Focus Techniques',
      description: 'Master techniques to achieve and maintain deep focus during study sessions.',
      duration: '25 min',
      difficulty: 'Beginner',
      status: 'New',
      completion: 0,
      recommended: true,
      mentalState: 'Focused',
      tags: ['Focus', 'Study Skills', 'Recommended'],
      brainFocus: ['Beta waves', 'Prefrontal cortex'],
      content: [
        "Introduction to focus techniques and their importance in learning",
        "Understanding the science behind focus and attention",
        "Technique 1: Pomodoro Method - Breaking work into focused intervals",
        "Technique 2: Mindful Focus - Bringing attention to the present task",
        "Technique 3: Environment Optimization for better concentration",
        "Practice exercises for developing sustained attention"
      ],
      objectives: [
        "Learn to maintain focus for extended periods",
        "Develop strategies to minimize distractions",
        "Understand how brainwaves correlate with focus states",
        "Practice techniques to enter flow state more consistently"
      ],
      prerequisites: [],
      relatedLessons: ["L2", "L5"]
    },
    {
      id: 'L2',
      title: 'Mindfulness for Learning',
      description: 'Apply mindfulness techniques to improve learning retention and reduce stress.',
      duration: '20 min',
      difficulty: 'Beginner',
      status: 'In Progress',
      completion: 45,
      recommended: true,
      mentalState: 'Relaxed',
      tags: ['Mindfulness', 'Stress Management', 'Recommended'],
      brainFocus: ['Alpha waves', 'Limbic system'],
      content: [
        "What is mindfulness and its benefits for learning",
        "The connection between relaxed states and information absorption",
        "Breathing techniques for entering relaxed alertness",
        "Mindful reading and listening practices",
        "Using mindfulness to enhance memory formation",
        "Integrating mindfulness into your daily study routine"
      ],
      objectives: [
        "Develop a regular mindfulness practice",
        "Learn to recognize and reduce mental stress during studying",
        "Enhance learning retention through relaxed awareness",
        "Improve ability to enter alpha brainwave states on demand"
      ],
      prerequisites: [],
      relatedLessons: ["L1", "L6"]
    },
    {
      id: 'L3',
      title: 'Memory Enhancement',
      description: 'Learn advanced memory techniques based on cognitive neuroscience principles.',
      duration: '30 min',
      difficulty: 'Intermediate',
      status: 'In Progress',
      completion: 70,
      recommended: false,
      mentalState: 'Focused',
      tags: ['Memory', 'Cognition'],
      brainFocus: ['Theta waves', 'Hippocampus'],
      content: [
        "Understanding memory formation: encoding, storage, retrieval",
        "The role of sleep and theta waves in memory consolidation",
        "Memory palace technique for enhanced information retention",
        "Spaced repetition systems for long-term memory",
        "Association techniques for complex concept learning",
        "Memory enhancement exercises and practical applications"
      ],
      objectives: [
        "Master at least two advanced memory techniques",
        "Understand how brainwave patterns affect memory formation",
        "Develop a personalized memory enhancement system",
        "Improve recall speed and accuracy for learned information"
      ],
      prerequisites: ["L1"],
      relatedLessons: ["L5"]
    },
    {
      id: 'L4',
      title: 'Sleep and Learning',
      description: 'Understand the critical relationship between sleep quality and cognitive performance.',
      duration: '15 min',
      difficulty: 'Beginner',
      status: 'Completed',
      completion: 100,
      recommended: false,
      mentalState: 'Drowsy',
      tags: ['Sleep', 'Recovery'],
      brainFocus: ['Delta waves', 'Brain stem'],
      content: [
        "The science of sleep cycles and learning",
        "How delta waves during deep sleep affect memory consolidation",
        "Optimizing sleep for better cognitive performance",
        "Pre-sleep learning techniques for enhanced retention",
        "Sleep hygiene practices for students",
        "Using brainwave monitoring to improve sleep quality"
      ],
      objectives: [
        "Understand the connection between sleep quality and learning efficiency",
        "Develop a sleep optimization routine for academic performance",
        "Learn to recognize signs of sleep deprivation affecting cognition",
        "Use techniques to improve deep sleep phases"
      ],
      prerequisites: [],
      relatedLessons: ["L6"]
    },
    {
      id: 'L5',
      title: 'Advanced Problem Solving',
      description: 'Develop neural pathways for enhanced analytical and creative problem solving.',
      duration: '45 min',
      difficulty: 'Advanced',
      status: 'New',
      completion: 0,
      recommended: true,
      mentalState: 'Focused',
      tags: ['Problem Solving', 'Critical Thinking', 'Recommended'],
      brainFocus: ['Gamma waves', 'Neocortex'],
      content: [
        "Understanding the neuroscience of problem-solving",
        "Gamma wave activity and its role in insight generation",
        "Analytical thinking frameworks for complex problems",
        "Creative thinking techniques to overcome mental blocks",
        "Combining divergent and convergent thinking methods",
        "Real-world problem-solving exercises and challenges"
      ],
      objectives: [
        "Develop advanced problem decomposition skills",
        "Learn to shift between creative and analytical thinking modes",
        "Master techniques to stimulate gamma wave activity",
        "Apply structured problem-solving frameworks to complex challenges"
      ],
      prerequisites: ["L1", "L3"],
      relatedLessons: ["L3"]
    },
    {
      id: 'L6',
      title: 'Stress Management',
      description: 'Techniques to manage stress and anxiety during learning and exams.',
      duration: '20 min',
      difficulty: 'Intermediate',
      status: 'Completed',
      completion: 100,
      recommended: false,
      mentalState: 'Relaxed',
      tags: ['Stress Management', 'Anxiety'],
      brainFocus: ['Alpha waves', 'Amygdala'],
      content: [
        "Understanding the stress response and its effect on learning",
        "The role of alpha waves in reducing anxiety",
        "Quick stress reduction techniques for exam situations",
        "Long-term stress management strategies for students",
        "Cognitive restructuring for academic anxiety",
        "Biofeedback methods using brainwave monitoring"
      ],
      objectives: [
        "Develop a toolkit of rapid stress reduction techniques",
        "Learn to increase alpha wave activity during high-pressure situations",
        "Create a personalized stress management plan",
        "Improve emotional regulation during learning challenges"
      ],
      prerequisites: ["L2"],
      relatedLessons: ["L2", "L4"]
    },
  ]);

  // Auto-update lessons based on mental state
  useEffect(() => {
    // Increase recommendation status for lessons matching current mental state
    const updatedLessons = lessons.map(lesson => {
      if (lesson.mentalState.toLowerCase() === mentalState.toLowerCase() && lesson.status !== 'Completed') {
        return {...lesson, recommended: true};
      }
      return lesson;
    });
    
    setLessons(updatedLessons);
  }, [mentalState]);

  // Filter lessons based on current tab and search term
  const filteredLessons = lessons.filter(lesson => {
    // First filter by tab selection
    const tabFilter = filter === 'all' || 
                     (filter === 'recommended' && lesson.recommended) || 
                     (lesson.status.toLowerCase().replace(' ', '-') === filter);
    
    // Then filter by search term
    const searchFilter = searchTerm === '' || 
                         lesson.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         lesson.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         lesson.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    
    return tabFilter && searchFilter;
  });

  // Get lessons that match current mental state
  const mentalStateRecommended = lessons.filter(
    lesson => lesson.mentalState.toLowerCase() === mentalState.toLowerCase()
  );

  // Status badge variant
  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'New': return 'default';
      case 'In Progress': return 'secondary';
      case 'Completed': return 'outline';
      default: return 'outline';
    }
  };

  // Difficulty badge variant
  const getDifficultyBadgeVariant = (difficulty: string) => {
    switch (difficulty) {
      case 'Beginner': return 'outline';
      case 'Intermediate': return 'secondary';
      case 'Advanced': return 'destructive';
      default: return 'outline';
    }
  };

  // Handle starting or continuing a lesson
  const handleLessonAction = (lesson: Lesson) => {
    setActiveLesson(lesson);
    setIsLessonOpen(true);
    
    // Update lesson status if it's new
    if (lesson.status === 'New') {
      const updatedLessons = lessons.map(l => {
        if (l.id === lesson.id) {
          return {...l, status: 'In Progress', completion: 5};
        }
        return l;
      });
      setLessons(updatedLessons);
      setLessonProgress(5);
      setCurrentSection(0);
    } else {
      setLessonProgress(lesson.completion);
      // Set current section based on completion
      if (lesson.content) {
        const sectionCount = lesson.content.length;
        setCurrentSection(Math.floor((lesson.completion / 100) * sectionCount));
      }
    }
  };

  // Handle lesson progress
  const handleLessonProgress = () => {
    if (!activeLesson || !activeLesson.content) return;
    
    const totalSections = activeLesson.content.length;
    const nextSection = Math.min(totalSections - 1, currentSection + 1);
    setCurrentSection(nextSection);
    
    // Calculate progress based on section
    const newProgress = Math.round(((nextSection + 1) / totalSections) * 100);
    
    // Update the lessons state
    const updatedLessons = lessons.map(l => {
      if (l.id === activeLesson.id) {
        const updatedStatus = newSection >= totalSections - 1 ? 'Completed' : 'In Progress';
        return {...l, status: updatedStatus, completion: newProgress};
      }
      return l;
    });
    
    setLessons(updatedLessons);
    setLessonProgress(newProgress);
    
    // Close dialog when completed
    if (nextSection >= totalSections - 1) {
      setTimeout(() => {
        setIsLessonOpen(false);
      }, 2000);
    }
  };

  return (
    <div className="container mx-auto space-y-8">
      <div className="flex justify-between items-center">
        <h2 className="text-3xl font-bold tracking-tight">Learning Recommendations</h2>
        <Badge variant="outline" className="px-3 py-1 text-base">
          Mental State: {mentalState}
        </Badge>
      </div>

      {/* Mental State Recommendations */}
      <Card className="bg-gradient-to-r from-indigo-50 to-purple-50 border-indigo-200">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-indigo-500" />
            Personalized for your current brain state
          </CardTitle>
          <CardDescription>
            Lessons optimized for your current mental state: <strong>{mentalState}</strong>
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {mentalStateRecommended.length > 0 ? (
              mentalStateRecommended.slice(0, 3).map(lesson => (
                <Card key={lesson.id} className="overflow-hidden hover:shadow-md transition-all">
                  <CardHeader className="p-4 pb-0">
                    <div className="flex justify-between">
                      <Badge variant={getStatusBadgeVariant(lesson.status)}>
                        {lesson.status}
                      </Badge>
                      <Badge variant="outline">
                        {lesson.duration}
                      </Badge>
                    </div>
                    <CardTitle className="mt-2 text-lg">{lesson.title}</CardTitle>
                  </CardHeader>
                  <CardContent className="p-4 pt-2">
                    <p className="text-sm text-muted-foreground">
                      {lesson.description}
                    </p>
                    
                    {lesson.status === 'In Progress' && (
                      <div className="mt-2">
                        <Progress value={lesson.completion} className="h-2" />
                        <p className="text-xs text-right text-muted-foreground mt-1">
                          {lesson.completion}% complete
                        </p>
                      </div>
                    )}
                  </CardContent>
                  <CardFooter className="p-4 pt-0 flex justify-between">
                    <Button 
                      variant="default" 
                      size="sm"
                      onClick={() => handleLessonAction(lesson)}
                    >
                      {lesson.status === 'New' ? 'Start Lesson' : 
                       lesson.status === 'In Progress' ? 'Continue' : 'Review'}
                    </Button>
                    <Badge variant={getDifficultyBadgeVariant(lesson.difficulty)}>
                      {lesson.difficulty}
                    </Badge>
                  </CardFooter>
                </Card>
              ))
            ) : (
              <div className="col-span-full text-center py-8">
                <p className="text-muted-foreground">No specific recommendations for your current mental state.</p>
                <p>Try exploring other categories below.</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Search and Filter */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
          <Input 
            placeholder="Search lessons..." 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>
        <Tabs defaultValue="all" value={filter} onValueChange={setFilter} className="w-full sm:w-auto">
          <TabsList className="grid grid-cols-5">
            <TabsTrigger value="all">All</TabsTrigger>
            <TabsTrigger value="recommended">Recommended</TabsTrigger>
            <TabsTrigger value="new">New</TabsTrigger>
            <TabsTrigger value="in-progress">In Progress</TabsTrigger>
            <TabsTrigger value="completed">Completed</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {/* All Lessons */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {filteredLessons.length > 0 ? (
          filteredLessons.map(lesson => (
            <Card 
              key={lesson.id} 
              className="overflow-hidden hover:shadow-md transition-all cursor-pointer"
              onClick={() => setSelectedLesson(lesson)}
            >
              <CardHeader className="p-4 pb-2">
                <div className="flex justify-between">
                  <Badge variant={getStatusBadgeVariant(lesson.status)}>
                    {lesson.status}
                  </Badge>
                  <div className="flex items-center gap-1 text-sm text-muted-foreground">
                    <Clock className="h-3 w-3" />
                    {lesson.duration}
                  </div>
                </div>
                <CardTitle className="mt-2">{lesson.title}</CardTitle>
              </CardHeader>
              <CardContent className="p-4 pt-2">
                <p className="text-sm text-muted-foreground mb-4">
                  {lesson.description}
                </p>
                
                {/* Brain focus areas */}
                <div className="mb-4">
                  <p className="text-xs text-muted-foreground mb-1">Brain Focus Areas:</p>
                  <div className="flex gap-1 flex-wrap">
                    {lesson.brainFocus.map(focus => (
                      <Badge key={focus} variant="outline" className="text-xs">
                        {focus}
                      </Badge>
                    ))}
                  </div>
                </div>
                
                {/* Progress bar for in-progress lessons */}
                {lesson.status === 'In Progress' && (
                  <div className="w-full bg-gray-200 rounded-full h-2 mb-1">
                    <div 
                      className="bg-indigo-600 h-2 rounded-full transition-all duration-1000" 
                      style={{ width: `${lesson.completion}%` }}
                    ></div>
                    <p className="text-xs text-right text-muted-foreground">
                      {lesson.completion}% complete
                    </p>
                  </div>
                )}
                
                {/* Completion award for completed lessons */}
                {lesson.status === 'Completed' && (
                  <div className="flex justify-center items-center text-green-500 mb-1">
                    <Award className="h-5 w-5 mr-1" />
                    <span className="text-sm font-medium">Completed</span>
                  </div>
                )}
              </CardContent>
              <CardFooter className="p-4 pt-0 flex justify-between items-center">
                <Button 
                  variant={lesson.status === 'Completed' ? "outline" : "default"} 
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleLessonAction(lesson);
                  }}
                >
                  {lesson.status === 'New' && (
                    <>
                      <BookOpen className="mr-1 h-4 w-4" />
                      Start Lesson
                    </>
                  )}
                  {lesson.status === 'In Progress' && (
                    <>
                      <BookOpen className="mr-1 h-4 w-4" />
                      Continue
                    </>
                  )}
                  {lesson.status === 'Completed' && (
                    <>
                      <CheckCircle className="mr-1 h-4 w-4" />
                      Review
                    </>
                  )}
                </Button>
                <Badge variant={getDifficultyBadgeVariant(lesson.difficulty)}>
                  {lesson.difficulty}
                </Badge>
              </CardFooter>
            </Card>
          ))
        ) : (
          <div className="col-span-full text-center py-12">
            <p className="text-lg text-muted-foreground">No lessons match your search criteria</p>
            <Button variant="outline" className="mt-4" onClick={() => {
              setSearchTerm('');
              setFilter('all');
            }}>
              Clear Filters
            </Button>
          </div>
        )}
      </div>

      {/* Lesson Detail Dialog */}
      {selectedLesson && (
        <Dialog open={!!selectedLesson} onOpenChange={(open) => !open && setSelectedLesson(null)}>
          <DialogContent className="max-w-3xl">
            <DialogHeader>
              <div className="flex items-center gap-2">
                <DialogTitle>{selectedLesson.title}</DialogTitle>
                <Badge variant={getStatusBadgeVariant(selectedLesson.status)}>
                  {selectedLesson.status}
                </Badge>
              </div>
              <DialogDescription>{selectedLesson.description}</DialogDescription>
            </DialogHeader>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium mb-1">Difficulty</h4>
                  <Badge variant={getDifficultyBadgeVariant(selectedLesson.difficulty)}>
                    {selectedLesson.difficulty}
                  </Badge>
                </div>
                <div>
                  <h4 className="font-medium mb-1">Duration</h4>
                  <div className="flex items-center">
                    <Clock className="h-4 w-4 mr-1" />
                    <span>{selectedLesson.duration}</span>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-medium mb-1">Learning Objectives</h4>
                <ul className="list-disc pl-5 space-y-1">
                  {selectedLesson.objectives?.map((objective, index) => (
                    <li key={index} className="text-sm">{objective}</li>
                  ))}
                </ul>
              </div>

              <div>
                <h4 className="font-medium mb-1">Content</h4>
                <ol className="list-decimal pl-5 space-y-1">
                  {selectedLesson.content?.map((item, index) => (
                    <li key={index} className="text-sm">{item}</li>
                  ))}
                </ol>
              </div>

              {selectedLesson.prerequisites && selectedLesson.prerequisites.length > 0 && (
                <div>
                  <h4 className="font-medium mb-1">Prerequisites</h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedLesson.prerequisites.map(prereqId => {
                      const prereq = lessons.find(l => l.id === prereqId);
                      return prereq ? (
                        <Badge key={prereqId} variant="outline">
                          {prereq.title}
                        </Badge>
                      ) : null;
                    })}
                  </div>
                </div>
              )}

              {selectedLesson.status === 'In Progress' && (
                <div>
                  <h4 className="font-medium mb-1">Progress</h4>
                  <Progress value={selectedLesson.completion} className="h-2" />
                  <p className="text-xs text-right text-muted-foreground mt-1">
                    {selectedLesson.completion}% complete
                  </p>
                </div>
              )}
            </div>
            
            <DialogFooter>
              <Button 
                variant="outline" 
                onClick={() => setSelectedLesson(null)}
              >
                Close
              </Button>
              <Button 
                onClick={() => {
                  handleLessonAction(selectedLesson);
                  setSelectedLesson(null);
                }}
              >
                {selectedLesson.status === 'New' ? 'Start Lesson' : 
                 selectedLesson.status === 'In Progress' ? 'Continue' : 'Review'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}

      {/* Active Lesson Dialog */}
      {activeLesson && (
        <Dialog open={isLessonOpen} onOpenChange={setIsLessonOpen} className="max-w-4xl">
          <DialogContent className="max-w-4xl">
            <DialogHeader>
              <DialogTitle className="flex items-center justify-between">
                <span>{activeLesson.title}</span>
                {lessonProgress >= 100 && (
                  <Badge variant="default" className="ml-2 animate-pulse">
                    Completed!
                  </Badge>
                )}
              </DialogTitle>
              <DialogDescription>
                {activeLesson.status === 'In Progress' ? 
                  `Continuing from where you left off (${lessonProgress}% complete)` : 
                  activeLesson.description
                }
              </DialogDescription>
            </DialogHeader>
            
            {/* Lesson Content */}
            <div className="space-y-6 my-6">
              {/* Progress bar */}
              <div className="mb-6">
                <div className="flex justify-between text-xs text-muted-foreground mb-1">
                  <span>Progress</span>
                  <span>{lessonProgress}%</span>
                </div>
                <Progress value={lessonProgress} className="h-2" />
              </div>
              
              {/* Current section content */}
              {activeLesson.content && currentSection < activeLesson.content.length && (
                <div className="bg-gray-50 p-6 rounded-lg border">
                  <div className="flex items-center mb-4">
                    <div className="bg-indigo-100 p-2 rounded-full">
                      <PlayCircle className="h-6 w-6 text-indigo-600" />
                    </div>
                    <h3 className="text-lg font-medium ml-2">
                      Section {currentSection + 1}: {activeLesson.content[currentSection].split(' - ')[0]}
                    </h3>
                  </div>
                  
                  <div className="space-y-4">
                    <p className="text-muted-foreground">
                      {activeLesson.content[currentSection].split(' - ')[1] || 
                       activeLesson.content[currentSection]}
                    </p>
                    
                    <div className="bg-white p-4 rounded-md border shadow-sm">
                      <h4 className="font-medium mb-2">Key Points</h4>
                      <ul className="list-disc pl-5 space-y-1">
                        <li>The {activeLesson.brainFocus[0]} plays a critical role in this technique</li>
                        <li>Regular practice strengthens neural pathways</li>
                        <li>Real-time EEG monitoring can help optimize your technique</li>
                      </ul>
                    </div>
                    
                    {/* Mock interactive elements */}
                    <div className="flex items-center space-x-4 mt-6">
                      <Avatar className="h-10 w-10">
                        <AvatarImage src="/images/Teacher.jpg" alt="Teacher" />
                        <AvatarFallback>TC</AvatarFallback>
                      </Avatar>
                      <div className="bg-blue-50 p-3 rounded-lg rounded-tl-none flex-1">
                        <p className="text-sm">
                          Remember that maintaining a consistent {activeLesson.mentalState.toLowerCase()} state 
                          while practicing these techniques will significantly enhance your results.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            <DialogFooter>
              <div className="flex items-center justify-between w-full">
                <Button 
                  variant="outline" 
                  onClick={() => setIsLessonOpen(false)}
                  disabled={lessonProgress === 100}
                >
                  Save Progress
                </Button>
                
                {lessonProgress < 100 ? (
                  <Button onClick={handleLessonProgress}>
                    Next Section <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                ) : (
                  <Button onClick={() => setIsLessonOpen(false)}>
                    Finish Lesson <CheckCircle className="ml-2 h-4 w-4" />
                  </Button>
                )}
              </div>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
};

export default LessonRecommendations;