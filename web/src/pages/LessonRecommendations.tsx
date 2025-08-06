import React, { useState, useEffect, useRef } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Clock,
  BookOpen,
  CheckCircle,
  Star,
  Award,
  Brain,
  Search,
  Filter,
  PlayCircle,
  ArrowRight,
  Video,
  Image,
  FileText,
  Activity,
  Target,
  Users,
  Zap,
  Puzzle,
  Palette,
  Music,
  Heart,
  Lightbulb,
  Gamepad2,
  Book,
  Calculator,
  Code,
  Globe,
  Target as TargetIcon,
  Sparkles,
  Coffee,
  Moon,
  Sun,
  Cloud,
} from "lucide-react";
import { useBrainData, MentalState } from "@/context/BrainDataContext";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { toast } from "@/hooks/use-toast";

interface LessonContent {
  type: "text" | "video" | "image" | "interactive" | "quiz" | "exercise";
  title: string;
  content: string;
  mediaUrl?: string;
  duration?: number;
  description?: string;
  keyPoints?: string[];
  interactiveElements?: {
    type: "button" | "slider" | "checkbox" | "input";
    label: string;
    options?: string[];
  }[];
}

interface Lesson {
  id: string;
  title: string;
  description: string;
  duration: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  status: "New" | "In Progress" | "Completed";
  completion: number;
  recommended: boolean;
  mentalState: string;
  tags: string[];
  brainFocus: string[];
  content: LessonContent[];
  objectives?: string[];
  prerequisites?: string[];
  relatedLessons?: string[];
  thumbnail?: string;
  instructor?: {
    name: string;
    avatar: string;
    credentials: string;
  };
  category: string;
  rating?: number;
  reviews?: number;
  lastUpdated?: string;
}

// Mental state activity mapping for children ages 10-18
const activityMap: Record<MentalState, {
  title: string;
  description: string;
  color: string;
  borderColor: string;
  icon: React.ReactNode;
  learn: Array<{ title: string; icon: React.ReactNode }>;
  mental: Array<{ title: string; icon: React.ReactNode }>;
}> = {
  Focused: {
    title: "🧠 Focused (Beta Waves)",
    description: "For kids deeply concentrating (e.g., during peak learning)",
    color: "from-blue-50 to-indigo-50",
    borderColor: "border-blue-200",
    icon: <Target className="h-6 w-6 text-blue-600" />,
    learn: [
      { title: "Solve 2 math puzzles or logic riddles", icon: <Puzzle className="h-4 w-4" /> },
      { title: "Work on a coding project (Scratch, Python basics, game dev)", icon: <Code className="h-4 w-4" /> },
      { title: "Finish science/computer textbook chapter with questions", icon: <Book className="h-4 w-4" /> },
      { title: "Play educational games (spelling bee, geography quiz)", icon: <Gamepad2 className="h-4 w-4" /> }
    ],
    mental: [
      { title: "25-minute focused session + 5-minute breathing break (Pomodoro)", icon: <Clock className="h-4 w-4" /> },
      { title: "Self-check: 'What did I just learn?'", icon: <Lightbulb className="h-4 w-4" /> },
      { title: "Celebrate success with stars or rewards", icon: <Star className="h-4 w-4" /> }
    ]
  },
  Relaxed: {
    title: "🌿 Relaxed (Alpha Waves)",
    description: "For kids who are calm and open to exploration",
    color: "from-green-50 to-emerald-50",
    borderColor: "border-green-200",
    icon: <Palette className="h-6 w-6 text-green-600" />,
    learn: [
      { title: "Create a drawing or diagram related to science/social studies", icon: <Palette className="h-4 w-4" /> },
      { title: "Read storybooks, history tales, or kid-friendly biographies", icon: <Book className="h-4 w-4" /> },
      { title: "Watch fun explainer videos (Why do we dream? How do airplanes fly?)", icon: <Video className="h-4 w-4" /> },
      { title: "Explore interactive simulations (math playground, pHET)", icon: <Globe className="h-4 w-4" /> }
    ],
    mental: [
      { title: "Calming background music while learning", icon: <Music className="h-4 w-4" /> },
      { title: "Coloring or doodling for a few minutes", icon: <Palette className="h-4 w-4" /> },
      { title: "3-minute guided visualization ('imagine your dream project...')", icon: <Sparkles className="h-4 w-4" /> }
    ]
  },
  Drowsy: {
    title: "😴 Drowsy (Delta/Theta Waves)",
    description: "For kids feeling tired, bored, or mentally saturated",
    color: "from-purple-50 to-violet-50",
    borderColor: "border-purple-200",
    icon: <Moon className="h-6 w-6 text-purple-600" />,
    learn: [
      { title: "Play a light memory or word game (crossword, word search)", icon: <Puzzle className="h-4 w-4" /> },
      { title: "Watch a short animated summary of previous topics", icon: <Video className="h-4 w-4" /> },
      { title: "Review flashcards or key points", icon: <FileText className="h-4 w-4" /> },
      { title: "Reflect: 'What did I like learning today?'", icon: <Heart className="h-4 w-4" /> }
    ],
    mental: [
      { title: "Do a simple breathing activity: Inhale 3 secs – Hold 2 – Exhale 4", icon: <Heart className="h-4 w-4" /> },
      { title: "Encourage rest or a brain break", icon: <Coffee className="h-4 w-4" /> },
      { title: "Suggest light stretching or walking", icon: <Activity className="h-4 w-4" /> }
    ]
  },
  Neutral: {
    title: "⚖️ Neutral (Balanced Waves)",
    description: "For kids neither fully focused nor tired — ready for mixed tasks",
    color: "from-gray-50 to-slate-50",
    borderColor: "border-gray-200",
    icon: <Cloud className="h-6 w-6 text-gray-600" />,
    learn: [
      { title: "Plan study goals for the day or week", icon: <TargetIcon className="h-4 w-4" /> },
      { title: "Mix of revision + discovery (review science + watch tech news)", icon: <Lightbulb className="h-4 w-4" /> },
      { title: "Choose your own activity (from a fun dashboard)", icon: <Gamepad2 className="h-4 w-4" /> },
      { title: "Do a quick quiz or self-evaluation", icon: <Calculator className="h-4 w-4" /> }
    ],
    mental: [
      { title: "Light movement breaks", icon: <Activity className="h-4 w-4" /> },
      { title: "Gratitude journaling (1 thing they liked today)", icon: <Heart className="h-4 w-4" /> },
      { title: "Mindfulness story (listen & reflect)", icon: <Book className="h-4 w-4" /> }
    ]
  }
};

const LessonRecommendations: React.FC = () => {
  const { mentalState } = useBrainData();
  const [filter, setFilter] = useState<string>("all");
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [selectedLesson, setSelectedLesson] = useState<Lesson | null>(null);
  const [activeLesson, setActiveLesson] = useState<Lesson | null>(null);
  const [isLessonOpen, setIsLessonOpen] = useState<boolean>(false);
  const [lessonProgress, setLessonProgress] = useState<number>(0);
  const [currentSection, setCurrentSection] = useState<number>(0);
  // Track previous mental state for auto-switch logic
  const prevMentalState = useRef<string>(mentalState);

  // Mock lesson data
  const [lessons, setLessons] = useState<Lesson[]>([
    {
      id: "L1",
      title: "Deep Focus Techniques",
      description:
        "Master techniques to achieve and maintain deep focus during study sessions.",
      duration: "25 min",
      difficulty: "Beginner",
      status: "New",
      completion: 0,
      recommended: true,
      mentalState: "Focused",
      tags: ["Focus", "Study Skills", "Recommended"],
      brainFocus: ["Beta waves", "Prefrontal cortex"],
      thumbnail:
        "https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=600",
      instructor: {
        name: "Dr. Sarah Chen",
        avatar:
          "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&w=150",
        credentials: "Neuroscientist, Stanford University",
      },
      category: "Focus & Concentration",
      rating: 4.8,
      reviews: 1247,
      lastUpdated: "2024-01-15",
      content: [
        {
          type: "text",
          title: "Introduction to Focus Techniques",
          content:
            "In this comprehensive introduction, we will explore the science behind focus and attention, and introduce you to the key techniques for achieving deep concentration. Understanding how your brain processes information and maintains attention is crucial for effective learning.",
          duration: 5,
          keyPoints: [
            "Understanding the neuroscience of attention",
            "Identifying common distractions and their impact",
            "The importance of a dedicated workspace",
            "How brainwaves correlate with focus states",
          ],
        },
        {
          type: "video",
          title: "Technique 1: Pomodoro Method",
          content:
            "Learn the Pomodoro Technique, a time management method that uses focused work intervals to maximize productivity and maintain concentration.",
          mediaUrl: "https://example.com/video1.mp4",
          duration: 10,
          keyPoints: [
            "25-minute focused work intervals",
            "5-minute breaks for mental recovery",
            "Longer breaks after 4 pomodoros",
            "Using timers to maintain discipline",
          ],
        },
        {
          type: "interactive",
          title: "Focus Assessment Exercise",
          content:
            "Take this interactive assessment to evaluate your current focus levels and identify areas for improvement.",
          duration: 8,
          interactiveElements: [
            {
              type: "slider",
              label: "Rate your current focus level (1-10)",
            },
            {
              type: "checkbox",
              label: "I get easily distracted by my phone",
            },
            {
              type: "checkbox",
              label: "I can maintain focus for 25+ minutes",
            },
            {
              type: "input",
              label: "What is your biggest distraction?",
            },
          ],
        },
        {
          type: "text",
          title: "Technique 2: Mindful Focus",
          content:
            "Mindful Focus involves bringing your attention to the present task and eliminating mental chatter. This technique combines mindfulness principles with cognitive training to enhance concentration.",
          duration: 5,
          keyPoints: [
            "Breathing techniques for focus",
            "Body scanning for tension release",
            "Thought observation without judgment",
            "Anchoring attention to the present moment",
          ],
        },
        {
          type: "video",
          title: "Technique 3: Environment Optimization",
          content:
            "Discover how to create an optimal learning environment that supports deep focus and minimizes distractions.",
          mediaUrl: "https://example.com/video2.mp4",
          duration: 10,
          keyPoints: [
            "Lighting and temperature optimization",
            "Noise management strategies",
            "Ergonomic workspace setup",
            "Digital environment organization",
          ],
        },
        {
          type: "exercise",
          title: "Focus Building Practice",
          content:
            "Practice exercises designed to strengthen your focus muscles and improve sustained attention.",
          duration: 7,
          keyPoints: [
            "Concentration meditation practice",
            "Single-tasking exercises",
            "Attention switching drills",
            "Focus endurance building",
          ],
        },
      ],
      objectives: [
        "Learn to maintain focus for extended periods",
        "Develop strategies to minimize distractions",
        "Understand how brainwaves correlate with focus states",
        "Practice techniques to enter flow state more consistently",
      ],
      prerequisites: [],
      relatedLessons: ["L2", "L5"],
    },
    {
      id: "L2",
      title: "Mindfulness for Learning",
      description:
        "Apply mindfulness techniques to improve learning retention and reduce stress.",
      duration: "20 min",
      difficulty: "Beginner",
      status: "In Progress",
      completion: 45,
      recommended: true,
      mentalState: "Relaxed",
      tags: ["Mindfulness", "Stress Management", "Recommended"],
      brainFocus: ["Alpha waves", "Limbic system"],
      thumbnail:
        "https://images.pexels.com/photos/3825586/pexels-photo-3825586.jpeg?auto=compress&cs=tinysrgb&w=600",
      instructor: {
        name: "Dr. Emily Watson",
        avatar:
          "https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?auto=compress&cs=tinysrgb&w=150",
        credentials: "Clinical Psychologist, Mindfulness Expert",
      },
      category: "Mindfulness & Wellness",
      rating: 4.9,
      reviews: 892,
      lastUpdated: "2024-01-10",
      content: [
        {
          type: "text",
          title: "What is Mindfulness?",
          content:
            "Mindfulness is the practice of paying attention to the present moment, without judgment. In this section, we will explore its benefits for learning and how to integrate it into your study routine.",
          duration: 5,
          keyPoints: [
            "Definition and core principles of mindfulness",
            "Benefits for learning and memory",
            "Common misconceptions about mindfulness",
            "Scientific evidence supporting mindfulness practice",
          ],
        },
        {
          type: "video",
          title: "Mindful Reading and Listening",
          content:
            "Learn how to apply mindfulness techniques to reading and listening activities to enhance comprehension and retention.",
          mediaUrl: "https://example.com/video3.mp4",
          duration: 10,
          keyPoints: [
            "Active reading strategies",
            "Mindful listening techniques",
            "Comprehension enhancement methods",
            "Retention improvement practices",
          ],
        },
        {
          type: "interactive",
          title: "Mindfulness Assessment",
          content:
            "Evaluate your current mindfulness practice and identify areas for growth.",
          duration: 6,
          interactiveElements: [
            {
              type: "slider",
              label: "How often do you practice mindfulness? (days per week)",
            },
            {
              type: "checkbox",
              label: "I can notice when my mind wanders",
            },
            {
              type: "checkbox",
              label: "I practice mindful breathing daily",
            },
            {
              type: "input",
              label: "What mindfulness technique interests you most?",
            },
          ],
        },
        {
          type: "text",
          title: "Using Mindfulness for Memory",
          content:
            "In this section, we will discuss how mindfulness can enhance memory formation and retention. We will explore the connection between relaxed states and information absorption.",
          duration: 5,
          keyPoints: [
            "Memory formation and mindfulness",
            "Stress reduction and learning",
            "Attention and memory connection",
            "Mindful study techniques",
          ],
        },
        {
          type: "video",
          title: "Integrating Mindfulness into Daily Study",
          content:
            "Discover practical ways to incorporate mindfulness into your daily study routine for better learning outcomes.",
          mediaUrl: "https://example.com/video4.mp4",
          duration: 10,
          keyPoints: [
            "Pre-study mindfulness rituals",
            "Mindful study breaks",
            "Post-study reflection practices",
            "Long-term mindfulness integration",
          ],
        },
      ],
      objectives: [
        "Develop a regular mindfulness practice",
        "Learn to recognize and reduce mental stress during studying",
        "Enhance learning retention through relaxed awareness",
        "Improve ability to enter alpha brainwave states on demand",
      ],
      prerequisites: [],
      relatedLessons: ["L1", "L6"],
    },
    {
      id: "L3",
      title: "Memory Enhancement",
      description:
        "Learn advanced memory techniques based on cognitive neuroscience principles.",
      duration: "30 min",
      difficulty: "Intermediate",
      status: "In Progress",
      completion: 70,
      recommended: false,
      mentalState: "Focused",
      tags: ["Memory", "Cognition"],
      brainFocus: ["Theta waves", "Hippocampus"],
      thumbnail:
        "https://www.verywellmind.com/thmb/sR1HibJwINIjO8q_M0JjMkuQMC0=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/171357703-56a792f23df78cf772974690.jpg",
      instructor: {
        name: "Dr. Marcus Rodriguez",
        avatar:
          "https://images.pexels.com/photos/1222271/pexels-photo-1222271.jpeg?auto=compress&cs=tinysrgb&w=150",
        credentials: "Cognitive Neuroscientist, MIT",
      },
      category: "Memory & Learning",
      rating: 4.7,
      reviews: 567,
      lastUpdated: "2024-01-08",
      content: [
        {
          type: "text",
          title: "Understanding Memory Formation",
          content:
            "In this comprehensive section, we will delve into the science of how memories are formed, stored, and retrieved. We will explore the role of sleep and theta waves in memory consolidation.",
          duration: 5,
          keyPoints: [
            "Memory encoding, storage, and retrieval processes",
            "The role of hippocampus in memory formation",
            "Theta waves and memory consolidation",
            "Sleep-dependent memory processing",
          ],
        },
        {
          type: "video",
          title: "Memory Palace Technique",
          content:
            "Master the ancient memory palace technique, a powerful method for enhancing information retention through spatial memory.",
          mediaUrl: "https://example.com/video5.mp4",
          duration: 12,
          keyPoints: [
            "Creating your first memory palace",
            "Associating information with locations",
            "Walking through your palace",
            "Strengthening memory associations",
          ],
        },
        {
          type: "interactive",
          title: "Memory Assessment",
          content:
            "Evaluate your current memory capabilities and identify areas for improvement.",
          duration: 8,
          interactiveElements: [
            {
              type: "slider",
              label: "Rate your memory for names and faces (1-10)",
            },
            {
              type: "checkbox",
              label: "I can remember phone numbers easily",
            },
            {
              type: "checkbox",
              label: "I use memory techniques regularly",
            },
            {
              type: "input",
              label: "What type of information do you struggle to remember?",
            },
          ],
        },
        {
          type: "text",
          title: "Spaced Repetition Systems",
          content:
            "In this section, we will discuss how spaced repetition systems can dramatically improve long-term memory retention. We will explore how association techniques can enhance complex concept learning.",
          duration: 5,
          keyPoints: [
            "The forgetting curve and spaced repetition",
            "Optimal timing for review sessions",
            "Digital spaced repetition tools",
            "Creating effective flashcards",
          ],
        },
        {
          type: "video",
          title: "Memory Enhancement Exercises",
          content:
            "Practice exercises designed to strengthen your memory and improve recall speed and accuracy.",
          mediaUrl: "https://example.com/video6.mp4",
          duration: 10,
          keyPoints: [
            "Association exercises for complex concepts",
            "Visual memory training",
            "Auditory memory enhancement",
            "Cross-modal memory integration",
          ],
        },
      ],
      objectives: [
        "Master at least two advanced memory techniques",
        "Understand how brainwave patterns affect memory formation",
        "Develop a personalized memory enhancement system",
        "Improve recall speed and accuracy for learned information",
      ],
      prerequisites: ["L1"],
      relatedLessons: ["L5"],
    },
    {
      id: "L4",
      title: "Sleep and Learning",
      description:
        "Understand the critical relationship between sleep quality and cognitive performance.",
      duration: "15 min",
      difficulty: "Beginner",
      status: "In Progress",
      completion: 0,
      recommended: false,
      mentalState: "Drowsy",
      tags: ["Sleep", "Recovery"],
      brainFocus: ["Delta waves", "Brain stem"],
      thumbnail:
        "https://images.pexels.com/photos/3771069/pexels-photo-3771069.jpeg?auto=compress&cs=tinysrgb&w=600",
      instructor: {
        name: "Dr. James Wilson",
        avatar:
          "https://images.pexels.com/photos/2379004/pexels-photo-2379004.jpeg?auto=compress&cs=tinysrgb&w=150",
        credentials: "Sleep Researcher, Harvard Medical School",
      },
      category: "Sleep & Recovery",
      rating: 4.6,
      reviews: 423,
      lastUpdated: "2024-01-05",
      content: [
        {
          type: "text",
          title: "The Science of Sleep and Learning",
          content:
            "In this comprehensive section, we will explore how sleep cycles and brainwave patterns affect cognitive performance. We will discuss how delta waves during deep sleep affect memory consolidation and how to optimize sleep for better cognitive performance.",
          duration: 5,
          keyPoints: [
            "Sleep cycles and their importance",
            "Delta waves and deep sleep",
            "Memory consolidation during sleep",
            "Sleep deprivation effects on cognition",
          ],
        },
        {
          type: "video",
          title: "Pre-Sleep Learning Techniques",
          content:
            "Discover techniques to optimize learning before sleep for enhanced memory consolidation.",
          mediaUrl: "https://example.com/video7.mp4",
          duration: 8,
          keyPoints: [
            "Optimal timing for pre-sleep study",
            "Content review strategies",
            "Relaxation techniques for better sleep",
            "Memory consolidation optimization",
          ],
        },
        {
          type: "interactive",
          title: "Sleep Quality Assessment",
          content:
            "Evaluate your current sleep habits and identify areas for improvement.",
          duration: 6,
          interactiveElements: [
            {
              type: "slider",
              label: "How many hours do you sleep per night?",
            },
            {
              type: "checkbox",
              label: "I have a consistent sleep schedule",
            },
            {
              type: "checkbox",
              label: "I avoid screens before bedtime",
            },
            {
              type: "input",
              label: "What is your biggest sleep challenge?",
            },
          ],
        },
        {
          type: "text",
          title: "Sleep Hygiene Practices",
          content:
            "In this section, we will discuss practical sleep hygiene practices that students can adopt to improve their academic performance. We will explore how to recognize signs of sleep deprivation affecting cognition and use techniques to improve deep sleep phases.",
          duration: 5,
          keyPoints: [
            "Creating a sleep-conducive environment",
            "Establishing a bedtime routine",
            "Managing stress before sleep",
            "Optimizing sleep timing",
          ],
        },
        {
          type: "video",
          title: "Using Brainwave Monitoring for Sleep",
          content:
            "Learn how to use brainwave monitoring to improve sleep quality and optimize learning.",
          mediaUrl: "https://example.com/video8.mp4",
          duration: 7,
          keyPoints: [
            "Understanding sleep brainwave patterns",
            "Monitoring sleep quality",
            "Optimizing sleep cycles",
            "Using data to improve sleep",
          ],
        },
      ],
      objectives: [
        "Understand the connection between sleep quality and learning efficiency",
        "Develop a sleep optimization routine for academic performance",
        "Learn to recognize signs of sleep deprivation affecting cognition",
        "Use techniques to improve deep sleep phases",
      ],
      prerequisites: [],
      relatedLessons: ["L6"],
    },
    {
      id: "L5",
      title: "Advanced Problem Solving",
      description:
        "Develop neural pathways for enhanced analytical and creative problem solving.",
      duration: "45 min",
      difficulty: "Advanced",
      status: "New",
      completion: 0,
      recommended: true,
      mentalState: "Focused",
      tags: ["Problem Solving", "Critical Thinking", "Recommended"],
      brainFocus: ["Gamma waves", "Neocortex"],
      thumbnail:
        "https://images.pexels.com/photos/1036623/pexels-photo-1036623.jpeg?auto=compress&cs=tinysrgb&w=600",
      instructor: {
        name: "Dr. Lisa Thompson",
        avatar:
          "https://images.pexels.com/photos/1036623/pexels-photo-1036623.jpeg?auto=compress&cs=tinysrgb&w=150",
        credentials: "Neuropsychologist, Cognitive Scientist",
      },
      category: "Problem Solving & Critical Thinking",
      rating: 4.9,
      reviews: 1500,
      lastUpdated: "2024-01-12",
      content: [
        {
          type: "text",
          title: "Understanding the Neuroscience of Problem-Solving",
          content:
            "In this comprehensive section, we will delve into the neuroscience of problem-solving, including how gamma wave activity and insight generation are linked. We will explore analytical thinking frameworks for complex problems and creative thinking techniques to overcome mental blocks.",
          duration: 5,
          keyPoints: [
            "Neuroscience of problem-solving",
            "Gamma wave activity and insight",
            "Analytical thinking frameworks",
            "Creative thinking techniques",
          ],
        },
        {
          type: "video",
          title: "Combining Divergent and Convergent Thinking",
          content:
            "Learn how to effectively combine divergent and convergent thinking to generate innovative solutions.",
          mediaUrl: "https://example.com/video9.mp4",
          duration: 10,
          keyPoints: [
            "Divergent thinking for creativity",
            "Convergent thinking for analysis",
            "Combining for optimal problem-solving",
            "Overcoming mental blocks",
          ],
        },
        {
          type: "interactive",
          title: "Problem-Solving Assessment",
          content:
            "Evaluate your current problem-solving skills and identify areas for improvement.",
          duration: 8,
          interactiveElements: [
            {
              type: "slider",
              label: "Rate your problem-solving ability (1-10)",
            },
            {
              type: "checkbox",
              label: "I struggle with abstract thinking",
            },
            {
              type: "checkbox",
              label: "I can quickly identify patterns",
            },
            {
              type: "input",
              label: "What type of problem do you find most challenging?",
            },
          ],
        },
        {
          type: "text",
          title: "Real-World Problem-Solving Exercises",
          content:
            "In this section, we will discuss how to apply structured problem-solving frameworks to complex challenges in real-world scenarios. We will explore how to develop advanced problem decomposition skills and learn to shift between creative and analytical thinking modes.",
          duration: 5,
          keyPoints: [
            "Structured problem-solving frameworks",
            "Advanced problem decomposition",
            "Creative and analytical thinking modes",
            "Applying to real-world problems",
          ],
        },
        {
          type: "video",
          title: "Stimulating Gamma Wave Activity",
          content:
            "Discover techniques to stimulate gamma wave activity for enhanced cognitive function and problem-solving.",
          mediaUrl: "https://example.com/video10.mp4",
          duration: 10,
          keyPoints: [
            "Gamma wave activity and its benefits",
            "Stimulation techniques",
            "Enhancing cognitive function",
            "Improving problem-solving",
          ],
        },
      ],
      objectives: [
        "Develop advanced problem decomposition skills",
        "Learn to shift between creative and analytical thinking modes",
        "Master techniques to stimulate gamma wave activity",
        "Apply structured problem-solving frameworks to complex challenges",
      ],
      prerequisites: ["L1", "L3"],
      relatedLessons: ["L3"],
    },
    {
      id: "L6",
      title: "Stress Management",
      description:
        "Techniques to manage stress and anxiety during learning and exams.",
      duration: "20 min",
      difficulty: "Intermediate",
      status: "Completed",
      completion: 100,
      recommended: false,
      mentalState: "Relaxed",
      tags: ["Stress Management", "Anxiety"],
      brainFocus: ["Alpha waves", "Amygdala"],
      thumbnail:
        "https://images.pexels.com/photos/1036623/pexels-photo-1036623.jpeg?auto=compress&cs=tinysrgb&w=600",
      instructor: {
        name: "Dr. Lisa Thompson",
        avatar:
          "https://images.pexels.com/photos/1036623/pexels-photo-1036623.jpeg?auto=compress&cs=tinysrgb&w=150",
        credentials: "Neuropsychologist, Cognitive Scientist",
      },
      category: "Problem Solving & Critical Thinking",
      rating: 4.9,
      reviews: 1500,
      lastUpdated: "2024-01-12",
      content: [
        {
          type: "text",
          title: "Understanding the Stress Response and its Effect on Learning",
          content:
            "In this comprehensive section, we will explore how the stress response and its effect on learning are linked. We will discuss how alpha waves can reduce anxiety and how quick stress reduction techniques can be beneficial in exam situations.",
          duration: 5,
          keyPoints: [
            "Stress response and its effect on learning",
            "Alpha waves and stress reduction",
            "Quick stress reduction techniques",
            "Exam stress management",
          ],
        },
        {
          type: "video",
          title: "Long-Term Stress Management Strategies",
          content:
            "Discover long-term strategies to manage stress and anxiety for sustained academic success.",
          mediaUrl: "https://example.com/video11.mp4",
          duration: 10,
          keyPoints: [
            "Long-term stress management",
            "Sustained academic success",
            "Emotional regulation",
            "Learning under pressure",
          ],
        },
        {
          type: "interactive",
          title: "Cognitive Restructuring for Academic Anxiety",
          content:
            "Learn how to use cognitive restructuring to manage academic anxiety and improve emotional regulation.",
          duration: 6,
          interactiveElements: [
            {
              type: "slider",
              label: "How anxious do you feel during exams? (1-10)",
            },
            {
              type: "checkbox",
              label: "I use relaxation techniques regularly",
            },
            {
              type: "checkbox",
              label: "I practice positive self-talk",
            },
            {
              type: "input",
              label: "What cognitive strategy helps you most?",
            },
          ],
        },
        {
          type: "video",
          title: "Biofeedback Methods Using Brainwave Monitoring",
          content:
            "Learn how to use brainwave monitoring as a biofeedback tool for stress reduction and emotional regulation.",
          mediaUrl: "https://example.com/video12.mp4",
          duration: 10,
          keyPoints: [
            "Biofeedback for stress reduction",
            "Emotional regulation",
            "Brainwave monitoring",
            "Learning under pressure",
          ],
        },
      ],
      objectives: [
        "Develop a toolkit of rapid stress reduction techniques",
        "Learn to increase alpha wave activity during high-pressure situations",
        "Create a personalized stress management plan",
        "Improve emotional regulation during learning challenges",
      ],
      prerequisites: ["L2"],
      relatedLessons: ["L2", "L4"],
    },
    // Entertainment lesson 1
    {
      id: "E1",
      title: "Quick Brain Game: Memory Match",
      description: "Boost your alertness with a fun memory matching game!",
      duration: "7 min",
      difficulty: "Beginner",
      status: "New",
      completion: 0,
      recommended: false,
      mentalState: "Drowsy",
      tags: ["Game", "Entertainment", "Fun"],
      brainFocus: ["Engagement", "Alertness"],
      thumbnail: "https://images.pexels.com/photos/442576/pexels-photo-442576.jpeg",
      instructor: {
        name: "Coach Alex Funster",
        avatar: "https://images.pexels.com/photos/91227/pexels-photo-91227.jpeg?auto=compress&cs=tinysrgb&w=150",
        credentials: "Gamification Specialist"
      },
      category: "Entertainment",
      rating: 4.7,
      reviews: 320,
      lastUpdated: "2024-03-10",
      content: [
        {
          type: "interactive",
          title: "Memory Match Game (Game)",
          content: "Flip the cards and try to match all pairs as quickly as possible!",
          duration: 7,
          keyPoints: [
            "Improves short-term memory",
            "Fun and engaging",
            "Quick break for your brain"
          ]
        }
      ],
      objectives: ["Increase alertness", "Have fun while learning"],
      prerequisites: [],
      relatedLessons: [],
    },
    // Entertainment lesson 2
    {
      id: "E2",
      title: "Desk Sports: Mini Basketball Challenge",
      description: "Get moving with a quick desk basketball game to re-energize!",
      duration: "5 min",
      difficulty: "Beginner",
      status: "New",
      completion: 0,
      recommended: false,
      mentalState: "Drowsy",
      tags: ["Sports", "Entertainment", "Movement"],
      brainFocus: ["Physical Activity", "Alertness"],
      thumbnail: "https://images.pexels.com/photos/33293092/pexels-photo-33293092.jpeg",
      instructor: {
        name: "Coach Jamie Move",
        avatar: "https://images.pexels.com/photos/1130626/pexels-photo-1130626.jpeg?auto=compress&cs=tinysrgb&w=150",
        credentials: "Physical Activity Coach"
      },
      category: "Entertainment",
      rating: 4.6,
      reviews: 210,
      lastUpdated: "2024-03-12",
      content: [
        {
          type: "exercise",
          title: "Mini Basketball Desk Game (Sports)",
          content: "Use a small ball and a cup to shoot hoops at your desk. Compete with yourself or a friend!",
          duration: 5,
          keyPoints: [
            "Physical movement boosts energy",
            "Fun competition",
            "Quick and easy to play"
          ]
        }
      ],
      objectives: ["Re-energize with movement", "Enjoy a quick sports break"],
      prerequisites: [],
      relatedLessons: [],
    },
    // Entertainment lesson 3
    {
      id: "E3",
      title: "Reaction Time Challenge",
      description: "Test and improve your reaction speed with this quick interactive game!",
      duration: "6 min",
      difficulty: "Beginner",
      status: "New",
      completion: 0,
      recommended: false,
      mentalState: "Drowsy",
      tags: ["Game", "Entertainment", "Reaction"],
      brainFocus: ["Alertness", "Speed"],
      thumbnail: "https://images.pexels.com/photos/6532370/pexels-photo-6532370.jpeg",
      instructor: {
        name: "Coach Quick Reflex",
        avatar: "https://images.pexels.com/photos/91227/pexels-photo-91227.jpeg?auto=compress&cs=tinysrgb&w=150",
        credentials: "Reflex Trainer"
      },
      category: "Entertainment",
      rating: 4.8,
      reviews: 150,
      lastUpdated: "2024-03-15",
      content: [
        {
          type: "interactive",
          title: "Reaction Time Game (Game)",
          content: "Click the button as soon as the screen changes color! Compete for your best time.",
          duration: 6,
          keyPoints: [
            "Improves alertness",
            "Fun and competitive",
            "Track your best score"
          ],
          interactiveElements: [
            { type: "button", label: "Start Game" }
          ]
        }
      ],
      objectives: ["Increase alertness", "Have fun while learning"],
      prerequisites: [],
      relatedLessons: [],
    },
    // Entertainment lesson 4
    {
      id: "E4",
      title: "Emoji Memory Blitz",
      description: "Remember the sequence of emojis and repeat it back!",
      duration: "8 min",
      difficulty: "Beginner",
      status: "New",
      completion: 0,
      recommended: false,
      mentalState: "Drowsy",
      tags: ["Game", "Entertainment", "Memory"],
      brainFocus: ["Memory", "Engagement"],
      thumbnail: "https://images.pexels.com/photos/207983/pexels-photo-207983.jpeg",
      instructor: {
        name: "Coach Emoji Fun",
        avatar: "https://images.pexels.com/photos/91227/pexels-photo-91227.jpeg?auto=compress&cs=tinysrgb&w=150",
        credentials: "Memory Coach"
      },
      category: "Entertainment",
      rating: 4.9,
      reviews: 180,
      lastUpdated: "2024-03-18",
      content: [
        {
          type: "interactive",
          title: "Emoji Memory Game (Game)",
          content: "Watch the emoji sequence, then repeat it by clicking the emojis in order!",
          duration: 8,
          keyPoints: [
            "Boosts working memory",
            "Colorful and fun",
            "Challenge yourself"
          ],
          interactiveElements: [
            { type: "button", label: "Start Game" }
          ]
        }
      ],
      objectives: ["Improve memory", "Have fun while drowsy"],
      prerequisites: [],
      relatedLessons: [],
    },
  ]);

  // Auto-update lessons based on mental state
  useEffect(() => {
    // Increase recommendation status for lessons matching current mental state
    const updatedLessons = lessons.map((lesson) => {
      if (
        lesson.mentalState.toLowerCase() === mentalState.toLowerCase() &&
        lesson.status !== "Completed"
      ) {
        return { ...lesson, recommended: true };
      }
      return lesson;
    });

    setLessons(updatedLessons);
  }, [mentalState]);

  // Filter lessons based on current tab and search term
  const filteredLessons = lessons.filter((lesson) => {
    // First filter by tab selection
    const tabFilter =
      filter === "all" ||
      (filter === "recommended" && lesson.recommended) ||
      lesson.status.toLowerCase().replace(" ", "-") === filter;

    // Then filter by search term
    const searchFilter =
      searchTerm === "" ||
      lesson.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      lesson.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
      lesson.tags.some((tag) =>
        tag.toLowerCase().includes(searchTerm.toLowerCase())
      );

    return tabFilter && searchFilter;
  });

  // Get lessons that match current mental state
  const mentalStateRecommended = lessons.filter(
    (lesson) => lesson.mentalState.toLowerCase() === mentalState.toLowerCase()
  );

  // Status badge variant
  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case "New":
        return "default";
      case "In Progress":
        return "secondary";
      case "Completed":
        return "outline";
      default:
        return "outline";
    }
  };

  // Difficulty badge variant
  const getDifficultyBadgeVariant = (difficulty: string) => {
    switch (difficulty) {
      case "Beginner":
        return "outline";
      case "Intermediate":
        return "secondary";
      case "Advanced":
        return "destructive";
      default:
        return "outline";
    }
  };

  // Handle starting or continuing a lesson
  const handleLessonAction = (lesson: Lesson) => {
    setActiveLesson(lesson);
    setIsLessonOpen(true);

    // Update lesson status if it's new
    if (lesson.status === "New") {
      const updatedLessons = lessons.map((l) => {
        if (l.id === lesson.id) {
          return { ...l, status: "In Progress" as const, completion: 5 };
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
    const updatedLessons = lessons.map((l) => {
      if (l.id === activeLesson.id) {
        const updatedStatus =
          nextSection >= totalSections - 1
            ? ("Completed" as const)
            : ("In Progress" as const);
        return { ...l, status: updatedStatus, completion: newProgress };
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

  useEffect(() => {
    // Dynamic live lesson switching based on mental state
    if (isLessonOpen && activeLesson && prevMentalState.current !== mentalState) {
      if (mentalState === "Drowsy") {
        // Find an entertainment/game lesson
        const entertainment = lessons.find(isEntertainmentLesson);
        if (entertainment && activeLesson.id !== entertainment.id) {
          setActiveLesson(entertainment);
          setCurrentSection(0);
          setLessonProgress(entertainment.completion || 0);
          toast({
            title: `Mental state changed to Drowsy`,
            description: `Switched to an interactive game/entertainment lesson to help you re-energize!`,
          });
        }
      } else if (mentalState === "Focused") {
        // Find a focused lesson
        const focused = lessons.find(
          (lesson) =>
            !isEntertainmentLesson(lesson) &&
            (lesson.mentalState === "Focused" || lesson.category.toLowerCase().includes("focus"))
        );
        if (focused && activeLesson.id !== focused.id) {
          setActiveLesson(focused);
          setCurrentSection(0);
          setLessonProgress(focused.completion || 0);
          toast({
            title: `Mental state changed to Focused`,
            description: `Switched to a focused study lesson to match your state!`,
          });
        }
      }
    }
    prevMentalState.current = mentalState;
  }, [mentalState, isLessonOpen, activeLesson, lessons]);

  // Entertainment lesson filter helper
  const isEntertainmentLesson = (lesson: Lesson) =>
    lesson.category.toLowerCase() === "entertainment" ||
    ["game", "sports", "entertainment"].some(type =>
      lesson.tags.map(t => t.toLowerCase()).includes(type) ||
      lesson.content.some(section =>
        (section.title && section.title.toLowerCase().includes(type))
      )
    );

  // Filter for entertainment lessons
  const entertainmentLessons = lessons.filter(isEntertainmentLesson);

  // Filter for focused study lessons
  const focusedLessons = lessons.filter(
    (lesson) =>
      !isEntertainmentLesson(lesson) &&
      (lesson.mentalState === "Focused" || lesson.category.toLowerCase().includes("focus"))
  );

  return (
    <div className="container mx-auto space-y-8">
      <div className="flex justify-between items-center">
        <h2 className="text-3xl font-bold tracking-tight">
          Learning Recommendations
        </h2>
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
            Lessons optimized for your current mental state:{" "}
            <strong>{mentalState}</strong>
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {mentalState === "Drowsy" ? (
              entertainmentLessons.length > 0 ? (
                entertainmentLessons.slice(0, 3).map((lesson) => (
                  <Card
                    key={lesson.id}
                    className="overflow-hidden hover:shadow-md transition-all"
                  >
                    <CardHeader className="p-4 pb-0">
                      <div className="flex justify-between">
                        <Badge variant={getStatusBadgeVariant(lesson.status)}>
                          {lesson.status}
                        </Badge>
                        <Badge variant="outline">{lesson.duration}</Badge>
                      </div>
                      <CardTitle className="mt-2 text-lg">
                        {lesson.title}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="p-4 pt-2">
                      <p className="text-sm text-muted-foreground">
                        {lesson.description}
                      </p>

                      {lesson.status === "In Progress" && (
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
                        {lesson.status === "New"
                          ? "Start Lesson"
                          : lesson.status === "In Progress"
                          ? "Continue"
                          : "Review"}
                      </Button>
                      <Badge
                        variant={getDifficultyBadgeVariant(lesson.difficulty)}
                      >
                        {lesson.difficulty}
                      </Badge>
                    </CardFooter>
                  </Card>
                ))
              ) : (
                <div className="col-span-full text-center py-8">
                  <p className="text-muted-foreground">
                    No entertainment lessons available for your current state.
                  </p>
                </div>
              )
            ) : (
              focusedLessons.length > 0 ? (
                focusedLessons.slice(0, 3).map((lesson) => (
                  <Card
                    key={lesson.id}
                    className="overflow-hidden hover:shadow-md transition-all"
                  >
                    <CardHeader className="p-4 pb-0">
                      <div className="flex justify-between">
                        <Badge variant={getStatusBadgeVariant(lesson.status)}>
                          {lesson.status}
                        </Badge>
                        <Badge variant="outline">{lesson.duration}</Badge>
                      </div>
                      <CardTitle className="mt-2 text-lg">
                        {lesson.title}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="p-4 pt-2">
                      <p className="text-sm text-muted-foreground">
                        {lesson.description}
                      </p>

                      {lesson.status === "In Progress" && (
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
                        {lesson.status === "New"
                          ? "Start Lesson"
                          : lesson.status === "In Progress"
                          ? "Continue"
                          : "Review"}
                      </Button>
                      <Badge
                        variant={getDifficultyBadgeVariant(lesson.difficulty)}
                      >
                        {lesson.difficulty}
                      </Badge>
                    </CardFooter>
                  </Card>
                ))
              ) : (
                <div className="col-span-full text-center py-8">
                  <p className="text-muted-foreground">
                    No focused study lessons available for your current state.
                  </p>
                </div>
              )
            )}
          </div>
        </CardContent>
      </Card>

      {/* Mental State-Based Learning & Mental Training (Ages 10-18) */}
      <Card className={`bg-gradient-to-r ${activityMap[mentalState]?.color || 'from-gray-50 to-slate-50'} ${activityMap[mentalState]?.borderColor || 'border-gray-200'}`}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {activityMap[mentalState]?.icon || <Brain className="h-5 w-5" />}
            {activityMap[mentalState]?.title || `Mental State: ${mentalState}`}
          </CardTitle>
          <CardDescription>
            {activityMap[mentalState]?.description || `Personalized activities for your current mental state: ${mentalState}`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 md:grid-cols-2">
            {/* Learning Activities */}
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <BookOpen className="h-5 w-5 text-blue-600" />
                <h3 className="text-lg font-semibold">🎓 Learning Activities</h3>
              </div>
              <div className="space-y-3">
                {activityMap[mentalState]?.learn.map((activity, index) => (
                  <Card key={index} className="bg-white/50 backdrop-blur-sm border-0 shadow-sm">
                    <CardContent className="p-4">
                      <div className="flex items-start gap-3">
                        <div className="flex-shrink-0 mt-0.5">
                          {activity.icon}
                        </div>
                        <p className="text-sm font-medium leading-relaxed">
                          {activity.title}
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>

            {/* Mental Training */}
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-purple-600" />
                <h3 className="text-lg font-semibold">🧘 Mental Training</h3>
              </div>
              <div className="space-y-3">
                {activityMap[mentalState]?.mental.map((activity, index) => (
                  <Card key={index} className="bg-white/50 backdrop-blur-sm border-0 shadow-sm">
                    <CardContent className="p-4">
                      <div className="flex items-start gap-3">
                        <div className="flex-shrink-0 mt-0.5">
                          {activity.icon}
                        </div>
                        <p className="text-sm font-medium leading-relaxed">
                          {activity.title}
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </div>

          {/* Quick Action Buttons */}
          <div className="mt-6 pt-6 border-t border-gray-200">
            <div className="flex flex-wrap gap-3">
              <Button variant="outline" className="flex items-center gap-2">
                <PlayCircle className="h-4 w-4" />
                Start Learning Session
              </Button>
              <Button variant="outline" className="flex items-center gap-2">
                <Activity className="h-4 w-4" />
                Begin Mental Training
              </Button>
              <Button variant="outline" className="flex items-center gap-2">
                <Target className="h-4 w-4" />
                Set Today's Goals
              </Button>
            </div>
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
        <Tabs
          defaultValue="all"
          value={filter}
          onValueChange={setFilter}
          className="w-full sm:w-auto"
        >
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
      <div className="grid gap-4 sm:gap-6 md:grid-cols-2 lg:grid-cols-3">
        {filteredLessons.length > 0 ? (
          filteredLessons.map(lesson => (
            <Card 
              key={lesson.id} 
              className="overflow-hidden hover:shadow-lg transition-all cursor-pointer group h-[500px] sm:h-[550px] md:h-[600px] flex flex-col"
              onClick={() => setSelectedLesson(lesson)}
            >
              {/* Lesson Thumbnail */}
              {lesson.thumbnail && (
                <div className="relative h-40 sm:h-48 overflow-hidden flex-shrink-0">
                  <img 
                    src={lesson.thumbnail} 
                    alt={lesson.title}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent" />
                  <div className="absolute top-2 sm:top-4 left-2 sm:left-4 flex items-center space-x-2">
                    <Badge variant={getStatusBadgeVariant(lesson.status)} className="text-xs">
                      {lesson.status}
                    </Badge>
                    {lesson.recommended && (
                      <Badge
                        variant="default"
                        className="bg-yellow-500 text-white text-xs"
                      >
                        <Star className="h-3 w-3 mr-1" />
                        Recommended
                      </Badge>
                    )}
                  </div>
                  <div className="absolute bottom-2 sm:bottom-4 right-2 sm:right-4 flex items-center space-x-1 text-white text-xs sm:text-sm">
                    <Clock className="h-3 w-3 sm:h-4 sm:w-4" />
                    <span>{lesson.duration}</span>
                  </div>
                </div>
              )}

              <CardHeader className="p-3 sm:p-4 pb-2 flex-shrink-0">
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <CardTitle className="text-base sm:text-lg line-clamp-2">
                      {lesson.title}
                    </CardTitle>
                    <CardDescription className="mt-2 line-clamp-2 text-xs sm:text-sm">
                      {lesson.description}
                    </CardDescription>
                  </div>
                </div>

                {/* Instructor Info */}
                {lesson.instructor && (
                  <div className="flex items-center space-x-2 mt-3">
                    <Avatar className="h-6 w-6 sm:h-8 sm:w-8 flex-shrink-0">
                      <AvatarImage
                        src={lesson.instructor.avatar}
                        alt={lesson.instructor.name}
                      />
                      <AvatarFallback className="text-xs">
                        {lesson.instructor.name
                          .split(" ")
                          .map((n) => n[0])
                          .join("")}
                      </AvatarFallback>
                    </Avatar>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs sm:text-sm font-medium truncate">
                        {lesson.instructor.name}
                      </p>
                      <p className="text-xs text-muted-foreground truncate">
                        {lesson.instructor.credentials}
                      </p>
                    </div>
                  </div>
                )}

                {/* Rating and Reviews */}
                {lesson.rating && (
                  <div className="flex items-center space-x-2 mt-2">
                    <div className="flex items-center">
                      {[...Array(5)].map((_, i) => (
                        <Star
                          key={i}
                          className={`h-3 w-3 sm:h-4 sm:w-4 ${
                            i < Math.floor(lesson.rating!)
                              ? "text-yellow-400 fill-current"
                              : "text-gray-300"
                          }`}
                        />
                      ))}
                    </div>
                    <span className="text-xs sm:text-sm text-muted-foreground">
                      {lesson.rating} ({lesson.reviews} reviews)
                    </span>
                  </div>
                )}
              </CardHeader>

              <CardContent className="p-3 sm:p-4 pt-2 flex-1 overflow-hidden">
                {/* Category and Tags */}
                <div className="mb-3 sm:mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <Badge variant="outline" className="text-xs">
                      {lesson.category}
                    </Badge>
                    <Badge
                      variant={getDifficultyBadgeVariant(lesson.difficulty)}
                      className="text-xs"
                    >
                      {lesson.difficulty}
                    </Badge>
                  </div>
                  
                  {/* Brain focus areas */}
                  <div className="mb-3 sm:mb-4">
                    <p className="text-xs text-muted-foreground mb-1">
                      Brain Focus Areas:
                    </p>
                    <div className="flex gap-1 flex-wrap">
                      {lesson.brainFocus.map((focus) => (
                        <Badge
                          key={focus}
                          variant="outline"
                          className="text-xs"
                        >
                          {focus}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
                
                {/* Progress bar for in-progress lessons */}
                {lesson.status === "In Progress" && (
                  <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
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
                {lesson.status === "Completed" && (
                  <div className="flex justify-center items-center text-green-500 mb-2">
                    <Award className="h-4 w-4 sm:h-5 sm:w-5 mr-1" />
                    <span className="text-xs sm:text-sm font-medium">Completed</span>
                  </div>
                )}

                {/* Content Preview - Limited to prevent overflow */}
                {/* <div className="mt-3 sm:mt-4">
                  <p className="text-xs text-muted-foreground mb-2">
                    Content Preview:
                  </p>
                  <div className="flex flex-wrap gap-1 max-h-16 sm:max-h-20 overflow-hidden">
                    {lesson.content.slice(0, 2).map((item, index) => (
                      <Badge key={index} variant="outline" className="text-xs">
                        {item.type === "video" && (
                          <Video className="h-3 w-3 mr-1" />
                        )}
                        {item.type === "interactive" && (
                          <Activity className="h-3 w-3 mr-1" />
                        )}
                        {item.type === "exercise" && (
                          <Target className="h-3 w-3 mr-1" />
                        )}
                        {item.type === "text" && (
                          <FileText className="h-3 w-3 mr-1" />
                        )}
                        {item.title.length > 15 ? `${item.title.substring(0, 15)}...` : item.title}
                      </Badge>
                    ))}
                    {lesson.content.length > 2 && (
                      <Badge variant="outline" className="text-xs">
                        +{lesson.content.length - 2} more
                      </Badge>
                    )}
                  </div>
                </div> */}
              </CardContent>

              <CardFooter className="p-3 sm:p-4 pt-0 flex justify-between items-center flex-shrink-0">
                <Button
                  variant={
                    lesson.status === "Completed" ? "outline" : "default"
                  }
                  size="sm"
                  className="text-xs sm:text-sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleLessonAction(lesson);
                  }}
                >
                  {lesson.status === "New" && (
                    <>
                      <BookOpen className="mr-1 h-3 w-3 sm:h-4 sm:w-4" />
                      Start Lesson
                    </>
                  )}
                  {lesson.status === "In Progress" && (
                    <>
                      <BookOpen className="mr-1 h-3 w-3 sm:h-4 sm:w-4" />
                      Continue
                    </>
                  )}
                  {lesson.status === "Completed" && (
                    <>
                      <CheckCircle className="mr-1 h-3 w-3 sm:h-4 sm:w-4" />
                      Review
                    </>
                  )}
                </Button>
                
                {/* Last Updated */}
                {lesson.lastUpdated && (
                  <span className="text-xs text-muted-foreground">
                    Updated {new Date(lesson.lastUpdated).toLocaleDateString()}
                  </span>
                )}
              </CardFooter>
            </Card>
          ))
        ) : (
          <div className="col-span-full text-center py-12">
            <p className="text-lg text-muted-foreground">
              No lessons match your search criteria
            </p>
            <Button
              variant="outline"
              className="mt-4"
              onClick={() => {
                setSearchTerm("");
                setFilter("all");
              }}
            >
              Clear Filters
            </Button>
          </div>
        )}
      </div>

      {/* Lesson Detail Dialog */}
      {selectedLesson && (
        <Dialog
          open={!!selectedLesson}
          onOpenChange={(open) => !open && setSelectedLesson(null)}
        >
          <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <div className="flex items-center gap-2">
                <DialogTitle>{selectedLesson.title}</DialogTitle>
                <Badge variant={getStatusBadgeVariant(selectedLesson.status)}>
                  {selectedLesson.status}
                </Badge>
              </div>
              <DialogDescription>
                {selectedLesson.description}
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-6">
              {/* Lesson Header Info */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="flex items-center space-x-2">
                  <Clock className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-sm font-medium">Duration</p>
                    <p className="text-sm text-muted-foreground">
                      {selectedLesson.duration}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Target className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-sm font-medium">Difficulty</p>
                    <Badge
                      variant={getDifficultyBadgeVariant(
                        selectedLesson.difficulty
                      )}
                    >
                      {selectedLesson.difficulty}
                    </Badge>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Users className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-sm font-medium">Category</p>
                    <p className="text-sm text-muted-foreground">
                      {selectedLesson.category}
                    </p>
                  </div>
                </div>
              </div>

              {/* Instructor Information */}
              {selectedLesson.instructor && (
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="font-medium mb-3">Instructor</h4>
                  <div className="flex items-center space-x-3">
                    <Avatar className="h-12 w-12">
                      <AvatarImage
                        src={selectedLesson.instructor.avatar}
                        alt={selectedLesson.instructor.name}
                      />
                      <AvatarFallback>
                        {selectedLesson.instructor.name
                          .split(" ")
                          .map((n) => n[0])
                          .join("")}
                      </AvatarFallback>
                    </Avatar>
                    <div>
                      <p className="font-medium">
                        {selectedLesson.instructor.name}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {selectedLesson.instructor.credentials}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Rating and Reviews */}
              {selectedLesson.rating && (
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="font-medium mb-3">Rating & Reviews</h4>
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center">
                      {[...Array(5)].map((_, i) => (
                        <Star
                          key={i}
                          className={`h-5 w-5 ${
                            i < Math.floor(selectedLesson.rating!)
                              ? "text-yellow-400 fill-current"
                              : "text-gray-300"
                          }`}
                        />
                      ))}
                    </div>
                    <div>
                      <p className="font-medium">
                        {selectedLesson.rating} out of 5
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {selectedLesson.reviews} reviews
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Learning Objectives */}
              {selectedLesson.objectives &&
                selectedLesson.objectives.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-3">Learning Objectives</h4>
                    <ul className="list-disc pl-5 space-y-2">
                      {selectedLesson.objectives.map((objective, index) => (
                        <li key={index} className="text-sm">
                          {objective}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

              {/* Content Overview */}
              <div>
                <h4 className="font-medium mb-3">Content Overview</h4>
                <div className="space-y-2">
                  {selectedLesson.content.map((item, index) => (
                    <div
                      key={index}
                      className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg"
                    >
                      <div className="flex-shrink-0">
                        {item.type === "video" && (
                          <Video className="h-4 w-4 text-blue-500" />
                        )}
                        {item.type === "interactive" && (
                          <Activity className="h-4 w-4 text-green-500" />
                        )}
                        {item.type === "exercise" && (
                          <Target className="h-4 w-4 text-purple-500" />
                        )}
                        {item.type === "text" && (
                          <FileText className="h-4 w-4 text-gray-500" />
                        )}
                        {item.type === "image" && (
                          <Image className="h-4 w-4 text-orange-500" />
                        )}
                      </div>
                      <div className="flex-1">
                        <p className="font-medium text-sm">{item.title}</p>
                        {item.duration && (
                          <p className="text-xs text-muted-foreground">
                            {item.duration} min
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Brain Focus Areas */}
              <div>
                <h4 className="font-medium mb-3">Brain Focus Areas</h4>
                <div className="flex flex-wrap gap-2">
                  {selectedLesson.brainFocus.map((focus) => (
                    <Badge key={focus} variant="outline">
                      {focus}
                    </Badge>
                  ))}
                </div>
              </div>

              {/* Prerequisites */}
              {selectedLesson.prerequisites &&
                selectedLesson.prerequisites.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-3">Prerequisites</h4>
                    <div className="flex flex-wrap gap-2">
                      {selectedLesson.prerequisites.map((prereqId) => {
                        const prereq = lessons.find((l) => l.id === prereqId);
                        return prereq ? (
                          <Badge key={prereqId} variant="outline">
                            {prereq.title}
                          </Badge>
                        ) : null;
                      })}
                    </div>
                  </div>
                )}

              {/* Progress for in-progress lessons */}
              {selectedLesson.status === "In Progress" && (
                <div>
                  <h4 className="font-medium mb-3">Progress</h4>
                  <Progress value={selectedLesson.completion} className="h-2" />
                  <p className="text-xs text-right text-muted-foreground mt-1">
                    {selectedLesson.completion}% complete
                  </p>
                </div>
              )}

              {/* Last Updated */}
              {selectedLesson.lastUpdated && (
                <div className="text-sm text-muted-foreground">
                  Last updated:{" "}
                  {new Date(selectedLesson.lastUpdated).toLocaleDateString()}
                </div>
              )}
            </div>

            <DialogFooter>
              <Button variant="outline" onClick={() => setSelectedLesson(null)}>
                Close
              </Button>
              <Button
                onClick={() => {
                  handleLessonAction(selectedLesson);
                  setSelectedLesson(null);
                }}
              >
                {selectedLesson.status === "New"
                  ? "Start Lesson"
                  : selectedLesson.status === "In Progress"
                  ? "Continue"
                  : "Review"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}

      {/* Active Lesson Dialog */}
      {activeLesson && (
        <Dialog open={isLessonOpen} onOpenChange={setIsLessonOpen}>
          <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
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
                {activeLesson.status === "In Progress"
                  ? `Continuing from where you left off (${lessonProgress}% complete)`
                  : activeLesson.description}
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
              {activeLesson.content &&
                currentSection < activeLesson.content.length && (
                  <div className="bg-gray-50 p-6 rounded-lg border">
                    <div className="flex items-center mb-4">
                      <div className="bg-indigo-100 p-2 rounded-full">
                        {activeLesson.content[currentSection].type ===
                          "video" && (
                          <Video className="h-6 w-6 text-indigo-600" />
                        )}
                        {activeLesson.content[currentSection].type ===
                          "interactive" && (
                          <Activity className="h-6 w-6 text-indigo-600" />
                        )}
                        {activeLesson.content[currentSection].type ===
                          "exercise" && (
                          <Target className="h-6 w-6 text-indigo-600" />
                        )}
                        {activeLesson.content[currentSection].type ===
                          "text" && (
                          <FileText className="h-6 w-6 text-indigo-600" />
                        )}
                        {activeLesson.content[currentSection].type ===
                          "image" && (
                          <Image className="h-6 w-6 text-indigo-600" />
                        )}
                      </div>
                      <h3 className="text-lg font-medium ml-2">
                        Section {currentSection + 1}:{" "}
                        {activeLesson.content[currentSection].title}
                      </h3>
                    </div>

                    <div className="space-y-4">
                      <p className="text-muted-foreground">
                        {activeLesson.content[currentSection].content}
                      </p>

                      {/* Media Content - Fixed height for videos */}
                      {activeLesson.content[currentSection].mediaUrl && (
                        <div className="relative w-full h-64 rounded-lg overflow-hidden">
                          {activeLesson.content[currentSection].type ===
                          "video" ? (
                            <video
                              src={
                                activeLesson.content[currentSection].mediaUrl
                              }
                              controls
                              className="w-full h-full object-cover"
                              poster={activeLesson.thumbnail}
                              preload="metadata"
                            />
                          ) : activeLesson.content[currentSection].type ===
                            "image" ? (
                            <img
                              src={
                                activeLesson.content[currentSection].mediaUrl
                              }
                              alt={activeLesson.content[currentSection].title}
                              className="w-full h-full object-cover"
                            />
                          ) : null}
                        </div>
                      )}

                      {/* Key Points */}
                      {activeLesson.content[currentSection].keyPoints &&
                        activeLesson.content[currentSection].keyPoints.length >
                          0 && (
                          <div className="bg-white p-4 rounded-md border shadow-sm">
                            <h4 className="font-medium mb-2">Key Points</h4>
                            <ul className="list-disc pl-5 space-y-1">
                              {activeLesson.content[
                                currentSection
                              ].keyPoints?.map((point, index) => (
                                <li key={index} className="text-sm">
                                  {point}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                      {/* Interactive Elements */}
                      {activeLesson.content[currentSection]
                        .interactiveElements &&
                        activeLesson.content[currentSection].interactiveElements
                          .length > 0 && (
                          <div className="bg-white p-4 rounded-md border shadow-sm">
                            <h4 className="font-medium mb-2">
                              Interactive Elements
                            </h4>
                            <div className="space-y-3">
                              {activeLesson.content[
                                currentSection
                              ].interactiveElements?.map((element, index) => (
                                <div
                                  key={index}
                                  className="flex items-center justify-between"
                                >
                                  <label className="text-sm font-medium">
                                    {element.label}
                                  </label>
                                  <div className="flex items-center space-x-2">
                                    {element.type === "button" && (
                                      <Button variant="outline" size="sm">
                                        {element.label}
                                      </Button>
                                    )}
                                    {element.type === "slider" && (
                                      <input
                                        type="range"
                                        min="0"
                                        max="100"
                                        defaultValue="50"
                                        className="w-32 h-2 rounded-lg accent-indigo-600"
                                      />
                                    )}
                                    {element.type === "checkbox" && (
                                      <input
                                        type="checkbox"
                                        className="h-4 w-4 accent-indigo-600"
                                      />
                                    )}
                                    {element.type === "input" && (
                                      <input
                                        type="text"
                                        placeholder="Enter your response..."
                                        className="w-48 h-8 px-2 rounded-md border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                                      />
                                    )}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                      {/* Instructor Tip */}
                      <div className="flex items-center space-x-4 mt-6">
                        <Avatar className="h-10 w-10 flex-shrink-0">
                          <AvatarImage
                            src={
                              activeLesson.instructor?.avatar ||
                              "/images/Teacher.jpg"
                            }
                            alt={activeLesson.instructor?.name || "Instructor"}
                          />
                          <AvatarFallback>
                            {activeLesson.instructor?.name
                              ?.split(" ")
                              .map((n) => n[0])
                              .join("") || "TC"}
                          </AvatarFallback>
                        </Avatar>
                        <div className="bg-blue-50 p-3 rounded-lg rounded-tl-none flex-1">
                          <p className="text-sm">
                            <strong>Instructor Tip:</strong> Remember that
                            maintaining a consistent{" "}
                            {activeLesson.mentalState.toLowerCase()} state while
                            practicing these techniques will significantly
                            enhance your results.
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
