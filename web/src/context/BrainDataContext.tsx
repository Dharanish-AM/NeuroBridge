import React, { createContext, useState, useContext, useEffect } from "react";

// Define types for our brain data
export interface ChannelData {
  name: string;
  value: number;
  history: number[];
}

export interface BrainwaveData {
  name: string;
  value: number; // Percentage
  color: string;
}

export type MentalState = "Focused" | "Relaxed" | "Drowsy" | "Neutral";

interface BrainDataContextType {
  channels: ChannelData[];
  brainwaves: BrainwaveData[];
  mentalState: MentalState;
  focusLevel: number; // 0-100
  timeSeriesData: { time: number; [key: string]: number }[];
  updateBrainData: () => void;
}

// Create context with a default value
const BrainDataContext = createContext<BrainDataContextType | undefined>(
  undefined
);

export const BrainDataProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [channels, setChannels] = useState<ChannelData[]>([]);
  const [brainwaves, setBrainwaves] = useState<BrainwaveData[]>([]);
  const [mentalState, setMentalState] = useState<MentalState>("Neutral");
  const [focusLevel, setFocusLevel] = useState<number>(0);
  const [timeSeriesData, setTimeSeriesData] = useState<
    { time: number; [key: string]: number }[]
  >([]);

  const updateBrainData = async () => {
    try {
      const testInputs = {
        Focused: { Fp1: 45, Fp2: 50, C3: 40, C4: 42 },
        Relaxed: { Fp1: 20, Fp2: 25, C3: 15, C4: 20 },
        Drowsy: { Fp1: -40, Fp2: -35, C3: -25, C4: -30 },
        Neutral: { Fp1: 5, Fp2: 5, C3: 5, C4: 5 },
      };
      const states = Object.keys(testInputs);
      const selectedState = states[
        Math.floor(Math.random() * states.length)
      ] as keyof typeof testInputs;
      const inputData = testInputs[selectedState];
      console.log("🧪 Selected test state:", selectedState, inputData);

      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputData),
      });
      const result = await response.json();
      console.log("Result :", result);
      const totalPower = Object.values(result as Record<string, number>).reduce(
        (acc, val) => acc + val,
        0
      );

      const brainwaveColors: Record<string, string> = {
        Delta: "#6366f1",
        Theta: "#8b5cf6",
        Alpha: "#ec4899",
        Beta: "#14b8a6",
        Gamma: "#f97316",
      };

      const brainwaves: BrainwaveData[] = Object.entries(result).map(
        ([name, value]) => {
          const properName =
            name.charAt(0).toUpperCase() + name.slice(1).toLowerCase();
          const numericValue =
            typeof value === "number" ? value : parseFloat(String(value));
          const percent = (numericValue / totalPower) * 100;
          return {
            name: properName,
            value: Math.round((percent + Number.EPSILON) * 100) / 100,
            color: brainwaveColors[properName] || "#000",
          };
        }
      );

      const dominantWave = brainwaves.reduce((a, b) =>
        a.value > b.value ? a : b
      );
      console.log(
        `🧠 Dominant: ${dominantWave.name} (${dominantWave.value.toFixed(2)}%)`
      );

      const mentalState: MentalState =
        dominantWave.name === "Beta"
          ? "Focused"
          : dominantWave.name === "Alpha"
          ? "Relaxed"
          : dominantWave.name === "Delta" || dominantWave.name === "Theta"
          ? "Drowsy"
          : "Neutral";

      const betaValue = result.Beta || 0;
      const alphaValue = result.Alpha || 0;
      const thetaValue = result.Theta || 0;

      const focusLevel = Math.min(
        100,
        Math.max(
          0,
          Math.round((betaValue / Math.max(1, thetaValue + alphaValue)) * 50)
        )
      );

      const dummyChannels: ChannelData[] = [
        { name: "Fp1", value: inputData.Fp1, history: [] },
        { name: "Fp2", value: inputData.Fp2, history: [] },
        { name: "C3", value: inputData.C3, history: [] },
        { name: "C4", value: inputData.C4, history: [] },
      ];

      const updatedChannels = dummyChannels.map((channel, index) => {
        const existingChannel = channels[index];
        const history = existingChannel
          ? [...existingChannel.history.slice(-50), channel.value]
          : [channel.value];
        return { ...channel, history };
      });

      const now = Date.now();
      const newTimePoint = {
        time: now,
        ...updatedChannels.reduce(
          (acc, channel) => ({ ...acc, [channel.name]: channel.value }),
          {}
        ),
      };

      setChannels(updatedChannels);
      setBrainwaves(brainwaves);
      setMentalState(mentalState);
      setFocusLevel(focusLevel);
      setTimeSeriesData((prev) => [...prev.slice(-100), newTimePoint]);
    } catch (error) {
      console.error("Error updating brain data:", error);
    }
  };

  // Initialize data on mount
  useEffect(() => {
    updateBrainData();
    // Set up interval for real-time updates
    const interval = setInterval(updateBrainData, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <BrainDataContext.Provider
      value={{
        channels,
        brainwaves,
        mentalState,
        focusLevel,
        timeSeriesData,
        updateBrainData,
      }}
    >
      {children}
    </BrainDataContext.Provider>
  );
};

// Custom hook to use the brain data context
export const useBrainData = () => {
  const context = useContext(BrainDataContext);
  if (context === undefined) {
    throw new Error("useBrainData must be used within a BrainDataProvider");
  }
  return context;
};
