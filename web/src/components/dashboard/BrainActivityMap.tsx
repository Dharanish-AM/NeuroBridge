import React, { useEffect, useRef, useState } from 'react';
import { useBrainData, ChannelData } from '@/context/BrainDataContext';

const BrainActivityMap: React.FC = () => {
  const { channels } = useBrainData();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  
  // Set up canvas dimensions based on container size
  useEffect(() => {
    const updateDimensions = () => {
      if (canvasRef.current && canvasRef.current.parentElement) {
        const { width } = canvasRef.current.parentElement.getBoundingClientRect();
        const height = Math.min(400, width * 0.6);
        setDimensions({ width, height });
      }
    };
    
    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);
  
  // Draw brain map when dimensions or channel data changes
  useEffect(() => {
    if (!canvasRef.current || dimensions.width === 0 || dimensions.height === 0) return;
    
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;
    
    // Set canvas dimensions
    canvasRef.current.width = dimensions.width;
    canvasRef.current.height = dimensions.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, dimensions.width, dimensions.height);
    
    // Draw brain outline
    drawBrainOutline(ctx, dimensions.width, dimensions.height);
    
    // Draw EEG electrode positions and activity levels
    drawElectrodePositions(ctx, channels, dimensions.width, dimensions.height);
    
  }, [dimensions, channels]);
  
  return (
    <div className="relative">
      <canvas 
        ref={canvasRef} 
        className="mx-auto"
      />
      <div className="absolute bottom-2 right-2 flex items-center gap-2">
        <div className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-blue-500 opacity-30"></span>
          <span className="text-xs text-muted-foreground">Low</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-blue-500 opacity-70"></span>
          <span className="text-xs text-muted-foreground">Medium</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-blue-500"></span>
          <span className="text-xs text-muted-foreground">High</span>
        </div>
      </div>
    </div>
  );
};

// Draw brain outline
const drawBrainOutline = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
  const centerX = width / 2;
  const centerY = height / 2;
  const brainWidth = width * 0.7;
  const brainHeight = height * 0.8;
  
  ctx.save();
  ctx.beginPath();
  
  // Draw a brain-like shape using bezier curves
  ctx.moveTo(centerX - brainWidth * 0.3, centerY - brainHeight * 0.1);
  ctx.bezierCurveTo(
    centerX - brainWidth * 0.4, centerY - brainHeight * 0.4,
    centerX - brainWidth * 0.5, centerY - brainHeight * 0.5,
    centerX - brainWidth * 0.3, centerY - brainHeight * 0.45
  );
  ctx.bezierCurveTo(
    centerX - brainWidth * 0.2, centerY - brainHeight * 0.5,
    centerX - brainWidth * 0.1, centerY - brainHeight * 0.5,
    centerX, centerY - brainHeight * 0.45
  );
  ctx.bezierCurveTo(
    centerX + brainWidth * 0.1, centerY - brainHeight * 0.5,
    centerX + brainWidth * 0.2, centerY - brainHeight * 0.5,
    centerX + brainWidth * 0.3, centerY - brainHeight * 0.45
  );
  ctx.bezierCurveTo(
    centerX + brainWidth * 0.5, centerY - brainHeight * 0.5,
    centerX + brainWidth * 0.4, centerY - brainHeight * 0.4,
    centerX + brainWidth * 0.3, centerY - brainHeight * 0.1
  );
  
  // Bottom part of the brain
  ctx.bezierCurveTo(
    centerX + brainWidth * 0.4, centerY + brainHeight * 0.2,
    centerX + brainWidth * 0.25, centerY + brainHeight * 0.4,
    centerX, centerY + brainHeight * 0.4
  );
  ctx.bezierCurveTo(
    centerX - brainWidth * 0.25, centerY + brainHeight * 0.4,
    centerX - brainWidth * 0.4, centerY + brainHeight * 0.2,
    centerX - brainWidth * 0.3, centerY - brainHeight * 0.1
  );
  
  // Draw center line to divide hemispheres
  ctx.moveTo(centerX, centerY - brainHeight * 0.45);
  ctx.lineTo(centerX, centerY + brainHeight * 0.4);
  
  ctx.strokeStyle = "#e2e8f0";  // Light gray
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
};

// Draw electrode positions and activity levels
const drawElectrodePositions = (
  ctx: CanvasRenderingContext2D, 
  channels: ChannelData[], 
  width: number, 
  height: number
) => {
  const centerX = width / 2;
  const centerY = height / 2;
  const brainWidth = width * 0.7;
  const brainHeight = height * 0.8;
  
  // Define electrode positions (normalized coordinates)
  const electrodePositions: Record<string, [number, number]> = {
    'Fp1': [-0.2, -0.4],  // Left Prefrontal
    'Fp2': [0.2, -0.4],   // Right Prefrontal
    'F7': [-0.4, -0.3],   // Left Frontal
    'F3': [-0.2, -0.25],  // Left Central Frontal
    'Fz': [0, -0.3],      // Midline Frontal
    'F4': [0.2, -0.25],   // Right Central Frontal
    'F8': [0.4, -0.3],    // Right Frontal
    'T3': [-0.45, 0],     // Left Temporal
    'C3': [-0.25, 0],     // Left Central
    'Cz': [0, 0],         // Midline Central
    'C4': [0.25, 0],      // Right Central
    'T4': [0.45, 0],      // Right Temporal
    'T5': [-0.4, 0.3],    // Left Posterior Temporal
    'P3': [-0.2, 0.25],   // Left Parietal
    'Pz': [0, 0.3],       // Midline Parietal
    'P4': [0.2, 0.25],    // Right Parietal
    'T6': [0.4, 0.3],     // Right Posterior Temporal
    'O1': [-0.2, 0.4],    // Left Occipital
    'O2': [0.2, 0.4]      // Right Occipital
  };
  
  // Draw all available channels
  channels.forEach(channel => {
    const position = electrodePositions[channel.name];
    
    // If we have a position for this electrode
    if (position) {
      const [xRatio, yRatio] = position;
      const x = centerX + xRatio * brainWidth;
      const y = centerY + yRatio * brainHeight;
      
      // Normalize the value for visualization (channel.value typically ranges from -100 to 100)
      const normalizedValue = Math.min(1, Math.max(0, (Math.abs(channel.value) / 100)));
      const radius = 6 + normalizedValue * 10;  // Size based on activity
      const alpha = 0.3 + normalizedValue * 0.7; // Opacity based on activity
      
      // Draw electrode circle
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(79, 70, 229, ${alpha})`;  // Indigo color with variable opacity
      ctx.fill();
      
      // Draw electrode label
      ctx.font = '10px sans-serif';
      ctx.fillStyle = '#64748b';  // Slate text color
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(channel.name, x, y + radius + 10);
    }
  });
  
  // Draw connections between adjacent electrodes to show propagation
  ctx.strokeStyle = 'rgba(79, 70, 229, 0.1)';  // Light indigo
  ctx.lineWidth = 1;
  
  const adjacentPairs = [
    ['Fp1', 'F3'], ['Fp2', 'F4'], ['F3', 'C3'], ['F4', 'C4'],
    ['C3', 'P3'], ['C4', 'P4'], ['P3', 'O1'], ['P4', 'O2'],
    ['Fp1', 'Fp2'], ['F3', 'Fz'], ['Fz', 'F4'], ['C3', 'Cz'],
    ['Cz', 'C4'], ['P3', 'Pz'], ['Pz', 'P4'], ['O1', 'O2']
  ];
  
  adjacentPairs.forEach(([ch1, ch2]) => {
    const pos1 = electrodePositions[ch1];
    const pos2 = electrodePositions[ch2];
    
    if (pos1 && pos2) {
      const x1 = centerX + pos1[0] * brainWidth;
      const y1 = centerY + pos1[1] * brainHeight;
      const x2 = centerX + pos2[0] * brainWidth;
      const y2 = centerY + pos2[1] * brainHeight;
      
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  });
};

export default BrainActivityMap;