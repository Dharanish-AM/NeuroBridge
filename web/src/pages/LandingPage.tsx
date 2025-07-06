import React, { useState, useEffect } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { 
  Brain, 
  Zap, 
  Target, 
  TrendingUp, 
  Users, 
  Award, 
  Play, 
  ChevronRight,
  CheckCircle,
  Star,
  ArrowRight,
  Activity,
  Shield,
  Sparkles,
  Globe,
  BookOpen,
  Timer,
  BarChart3
} from 'lucide-react';

const LandingPage: React.FC = () => {
  const [activeFeature, setActiveFeature] = useState(0);
  const { scrollY } = useScroll();
  const y1 = useTransform(scrollY, [0, 300], [0, 50]);
  const y2 = useTransform(scrollY, [0, 300], [0, -50]);

  const features = [
    {
      icon: Activity,
      title: "Real-time EEG Monitoring",
      description: "Advanced brainwave visualization with live mental state analysis",
      image: "https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=600"
    },
    {
      icon: Target,
      title: "Personalized Training",
      description: "AI-powered cognitive exercises that adapt to your brain patterns",
      image: "https://images.pexels.com/photos/3825586/pexels-photo-3825586.jpeg?auto=compress&cs=tinysrgb&w=600"
    },
    {
      icon: BarChart3,
      title: "Progress Analytics",
      description: "Comprehensive reports tracking your mental performance over time",
      image: "https://images.pexels.com/photos/590022/pexels-photo-590022.jpg?auto=compress&cs=tinysrgb&w=600"
    }
  ];

  const testimonials = [
    {
      name: "Dr. Sarah Chen",
      role: "Neuroscientist, Stanford University",
      content: "NeuroBridge represents a breakthrough in accessible neurofeedback technology. The real-time analysis is remarkably accurate.",
      avatar: "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&w=150",
      rating: 5
    },
    {
      name: "Marcus Rodriguez",
      role: "Software Engineer",
      content: "As someone with ADHD, NeuroBridge has transformed my ability to focus. The training modules are engaging and effective.",
      avatar: "https://images.pexels.com/photos/1222271/pexels-photo-1222271.jpeg?auto=compress&cs=tinysrgb&w=150",
      rating: 5
    },
    {
      name: "Dr. Emily Watson",
      role: "Clinical Psychologist",
      content: "I recommend NeuroBridge to my patients. The meditation guidance and stress reduction features are exceptional.",
      avatar: "https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?auto=compress&cs=tinysrgb&w=150",
      rating: 5
    }
  ];

  const stats = [
    { number: "50K+", label: "Active Users", icon: Users },
    { number: "2M+", label: "Training Sessions", icon: Target },
    { number: "94%", label: "Improvement Rate", icon: TrendingUp },
    { number: "15+", label: "Research Partners", icon: Award }
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveFeature((prev) => (prev + 1) % features.length);
    }, 4000);
    return () => clearInterval(interval);
  }, [features.length]);

  return (
    <div className="min-h-screen bg-white dark:bg-gray-900 overflow-hidden">
      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center px-4 sm:px-6 lg:px-8">
        {/* Background Elements */}
        <div className="absolute inset-0 overflow-hidden">
          <motion.div
            style={{ y: y1 }}
            className="absolute top-20 left-10 w-72 h-72 bg-blue-500/10 rounded-full blur-3xl"
          />
          <motion.div
            style={{ y: y2 }}
            className="absolute bottom-20 right-10 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"
          />
          <div className="absolute inset-0 bg-gradient-to-br from-blue-50/50 to-purple-50/50 dark:from-blue-900/20 dark:to-purple-900/20" />
        </div>

        <div className="relative max-w-7xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-8"
          >
            <div className="inline-flex items-center space-x-2 bg-blue-100 dark:bg-blue-900/20 text-blue-800 dark:text-blue-300 px-4 py-2 rounded-full text-sm font-medium mb-6">
              <Sparkles className="h-4 w-4" />
              <span>Revolutionary Brain Training Platform</span>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold text-gray-900 dark:text-white mb-6 leading-tight">
              Unlock Your
              <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent"> Mind's </span>
              Potential
            </h1>
            
            <p className="text-xl md:text-2xl text-gray-600 dark:text-gray-300 mb-8 max-w-3xl mx-auto leading-relaxed">
              Advanced EEG monitoring and AI-powered training to enhance focus, reduce stress, 
              and optimize your cognitive performance in real-time.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6 mb-12"
          >
            <button className="group bg-blue-600 text-white px-8 py-4 rounded-xl font-semibold text-lg hover:bg-blue-700 transition-all duration-300 flex items-center space-x-2 shadow-lg hover:shadow-xl transform hover:scale-105">
              <span>Start Free Trial</span>
              <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
            </button>
            
            <button className="group flex items-center space-x-3 text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
              <div className="w-12 h-12 bg-white dark:bg-gray-800 rounded-full flex items-center justify-center shadow-lg group-hover:shadow-xl transition-shadow">
                <Play className="h-5 w-5 ml-1" />
              </div>
              <span className="font-medium">Watch Demo</span>
            </button>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto"
          >
            {stats.map((stat, index) => {
              const Icon = stat.icon;
              return (
                <div key={index} className="text-center">
                  <div className="inline-flex items-center justify-center w-12 h-12 bg-blue-100 dark:bg-blue-900/20 rounded-lg mb-3">
                    <Icon className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                  </div>
                  <div className="text-3xl font-bold text-gray-900 dark:text-white mb-1">
                    {stat.number}
                  </div>
                  <div className="text-gray-600 dark:text-gray-400 text-sm">
                    {stat.label}
                  </div>
                </div>
              );
            })}
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 bg-gray-50 dark:bg-gray-800/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-6">
              Cutting-Edge Features
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              Experience the future of brain training with our scientifically-backed platform
            </p>
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            {/* Feature List */}
            <div className="space-y-8">
              {features.map((feature, index) => {
                const Icon = feature.icon;
                return (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -30 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.6, delay: index * 0.1 }}
                    viewport={{ once: true }}
                    className={`p-6 rounded-2xl cursor-pointer transition-all duration-300 ${
                      activeFeature === index
                        ? 'bg-blue-100 dark:bg-blue-900/20 border-2 border-blue-300 dark:border-blue-600'
                        : 'bg-white dark:bg-gray-800 border-2 border-transparent hover:border-gray-200 dark:hover:border-gray-700'
                    }`}
                    onClick={() => setActiveFeature(index)}
                  >
                    <div className="flex items-start space-x-4">
                      <div className={`p-3 rounded-xl ${
                        activeFeature === index
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
                      }`}>
                        <Icon className="h-6 w-6" />
                      </div>
                      <div>
                        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                          {feature.title}
                        </h3>
                        <p className="text-gray-600 dark:text-gray-300">
                          {feature.description}
                        </p>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>

            {/* Feature Image */}
            <motion.div
              key={activeFeature}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
              className="relative"
            >
              <div className="relative rounded-2xl overflow-hidden shadow-2xl">
                <img
                  src={features[activeFeature].image}
                  alt={features[activeFeature].title}
                  className="w-full h-96 object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent" />
                <div className="absolute bottom-6 left-6 text-white">
                  <h4 className="text-xl font-semibold mb-2">
                    {features[activeFeature].title}
                  </h4>
                  <p className="text-white/90">
                    {features[activeFeature].description}
                  </p>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-6">
              How NeuroBridge Works
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              Simple steps to transform your mental performance
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                step: "01",
                title: "Connect Your EEG Device",
                description: "Compatible with popular EEG headsets like Muse, Emotiv, and OpenBCI",
                icon: Brain,
                color: "blue"
              },
              {
                step: "02",
                title: "Real-time Analysis",
                description: "Our AI analyzes your brainwaves and provides instant feedback on your mental state",
                icon: Zap,
                color: "purple"
              },
              {
                step: "03",
                title: "Personalized Training",
                description: "Adaptive exercises and recommendations based on your unique brain patterns",
                icon: Target,
                color: "green"
              }
            ].map((item, index) => {
              const Icon = item.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.2 }}
                  viewport={{ once: true }}
                  className="relative text-center"
                >
                  <div className="relative mb-8">
                    <div className={`w-20 h-20 mx-auto rounded-full bg-${item.color}-100 dark:bg-${item.color}-900/20 flex items-center justify-center mb-4`}>
                      <Icon className={`h-10 w-10 text-${item.color}-600 dark:text-${item.color}-400`} />
                    </div>
                    <div className={`absolute -top-2 -right-2 w-8 h-8 bg-${item.color}-600 text-white rounded-full flex items-center justify-center text-sm font-bold`}>
                      {item.step}
                    </div>
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                    {item.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300">
                    {item.description}
                  </p>
                  {index < 2 && (
                    <div className="hidden md:block absolute top-10 left-full w-full">
                      <ChevronRight className="h-6 w-6 text-gray-400 mx-auto" />
                    </div>
                  )}
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="py-24 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/10 dark:to-purple-900/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-6">
              Transform Your Mind
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              Scientifically proven benefits for your cognitive health and performance
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                icon: Target,
                title: "Enhanced Focus",
                description: "Improve concentration and attention span by up to 40% with targeted training",
                stat: "40%",
                color: "blue"
              },
              {
                icon: Shield,
                title: "Stress Reduction",
                description: "Lower cortisol levels and achieve deeper relaxation states",
                stat: "65%",
                color: "green"
              },
              {
                icon: BookOpen,
                title: "Better Learning",
                description: "Accelerate information retention and cognitive processing",
                stat: "30%",
                color: "purple"
              },
              {
                icon: Timer,
                title: "Improved Sleep",
                description: "Optimize brainwave patterns for better sleep quality",
                stat: "50%",
                color: "indigo"
              },
              {
                icon: TrendingUp,
                title: "Peak Performance",
                description: "Achieve flow states and optimal mental performance",
                stat: "85%",
                color: "orange"
              },
              {
                icon: Brain,
                title: "Cognitive Health",
                description: "Support long-term brain health and neuroplasticity",
                stat: "95%",
                color: "pink"
              }
            ].map((benefit, index) => {
              const Icon = benefit.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="bg-white dark:bg-gray-800 p-8 rounded-2xl shadow-lg hover:shadow-xl transition-shadow"
                >
                  <div className={`w-16 h-16 bg-${benefit.color}-100 dark:bg-${benefit.color}-900/20 rounded-xl flex items-center justify-center mb-6`}>
                    <Icon className={`h-8 w-8 text-${benefit.color}-600 dark:text-${benefit.color}-400`} />
                  </div>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                      {benefit.title}
                    </h3>
                    <span className={`text-2xl font-bold text-${benefit.color}-600 dark:text-${benefit.color}-400`}>
                      +{benefit.stat}
                    </span>
                  </div>
                  <p className="text-gray-600 dark:text-gray-300">
                    {benefit.description}
                  </p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-6">
              Trusted by Experts
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              See what researchers, professionals, and users say about NeuroBridge
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-white dark:bg-gray-800 p-8 rounded-2xl shadow-lg"
              >
                <div className="flex items-center space-x-1 mb-4">
                  {[...Array(testimonial.rating)].map((_, i) => (
                    <Star key={i} className="h-5 w-5 text-yellow-400 fill-current" />
                  ))}
                </div>
                <p className="text-gray-600 dark:text-gray-300 mb-6 italic">
                  "{testimonial.content}"
                </p>
                <div className="flex items-center space-x-4">
                  <img
                    src={testimonial.avatar}
                    alt={testimonial.name}
                    className="w-12 h-12 rounded-full object-cover"
                  />
                  <div>
                    <div className="font-semibold text-gray-900 dark:text-white">
                      {testimonial.name}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      {testimonial.role}
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-gradient-to-r from-blue-600 to-purple-600">
        <div className="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Ready to Unlock Your Potential?
            </h2>
            <p className="text-xl text-blue-100 mb-8 max-w-2xl mx-auto">
              Join thousands of users who have transformed their mental performance with NeuroBridge
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6">
              <button className="bg-white text-blue-600 px-8 py-4 rounded-xl font-semibold text-lg hover:bg-gray-100 transition-colors shadow-lg">
                Start Free Trial
              </button>
              <button className="text-white border-2 border-white px-8 py-4 rounded-xl font-semibold text-lg hover:bg-white hover:text-blue-600 transition-colors">
                Schedule Demo
              </button>
            </div>
            <p className="text-blue-100 text-sm mt-6">
              No credit card required • 14-day free trial • Cancel anytime
            </p>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="col-span-1 md:col-span-2">
              <div className="flex items-center space-x-3 mb-6">
                <Brain className="h-8 w-8 text-blue-400" />
                <span className="text-2xl font-bold">NeuroBridge</span>
              </div>
              <p className="text-gray-400 mb-6 max-w-md">
                Revolutionizing brain training through advanced EEG technology and AI-powered insights.
              </p>
              <div className="flex items-center space-x-4">
                <Globe className="h-5 w-5 text-gray-400" />
                <span className="text-gray-400">Available worldwide</span>
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">Features</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Pricing</a></li>
                <li><a href="#" className="hover:text-white transition-colors">API</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Integrations</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">About</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Research</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Careers</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Contact</a></li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-gray-800 mt-12 pt-8 flex flex-col md:flex-row items-center justify-between">
            <p className="text-gray-400 text-sm">
              © 2024 NeuroBridge. All rights reserved.
            </p>
            <div className="flex items-center space-x-6 mt-4 md:mt-0">
              <a href="#" className="text-gray-400 hover:text-white transition-colors text-sm">Privacy</a>
              <a href="#" className="text-gray-400 hover:text-white transition-colors text-sm">Terms</a>
              <a href="#" className="text-gray-400 hover:text-white transition-colors text-sm">Security</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;