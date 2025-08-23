import React, { useState, useEffect, useRef } from 'react';
import { Brain, Zap, AlertTriangle, CheckCircle, StopCircle, MemoryStick, Clock } from 'lucide-react';
import { cn } from '@/lib/utils';

interface TrainPageProps {
  appState: {
    totalDuration: number;
    avgQuality: number;
    clipCount: number;
  };
  onComplete: () => void;
}

interface TrainingStatus {
  status: string;
  progress_pct: number;
  eta_min: number;
  current_epoch: number;
  mode: string;
  error?: string;
}

export function TrainPage({ appState, onComplete }: TrainPageProps) {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [error, setError] = useState('');
  const [memoryUsage, setMemoryUsage] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    // Check existing training status
    checkTrainingStatus();
    
    // Monitor memory usage
    const memoryInterval = setInterval(fetchMemoryUsage, 5000);
    
    return () => {
      clearInterval(memoryInterval);
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const checkTrainingStatus = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/train/status');
      const status = await response.json();
      
      if (status.status === 'training') {
        setTrainingStatus(status);
        setIsTraining(true);
        startProgressMonitoring();
      } else if (status.status === 'completed') {
        setTrainingStatus(status);
        addLog('Training completed successfully!');
      }
    } catch (err) {
      console.error('Failed to check training status:', err);
    }
  };

  const fetchMemoryUsage = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/healthz');
      const health = await response.json();
      setMemoryUsage(health.memory_usage_gb);
      
      // Emergency stop if memory exceeds 7GB
      if (health.memory_usage_gb > 7 && isTraining) {
        addLog('🚨 Emergency stop: Memory usage exceeded 7GB');
        stopTraining();
      }
    } catch (err) {
      console.error('Failed to fetch memory usage:', err);
    }
  };

  const startTraining = async () => {
    setIsTraining(true);
    setError('');
    setLogs([]);
    addLog('Starting AI training...');
    
    try {
      const response = await fetch('http://127.0.0.1:8000/train', {
        method: 'POST'
      });
      
      const result = await response.json();
      
      if (response.ok) {
        addLog(`Training started in ${result.mode} mode`);
        addLog(`Estimated time: ${result.estimated_time_min} minutes`);
        addLog(`Processing ${result.dataset_size} clips`);
        
        startProgressMonitoring();
      } else {
        setError(result.detail || 'Failed to start training');
        setIsTraining(false);
        addLog(`❌ Error: ${result.detail}`);
      }
    } catch (err) {
      setError('Network error during training start');
      setIsTraining(false);
      addLog('❌ Network error');
    }
  };

  const startProgressMonitoring = () => {
    // Close existing EventSource
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    // Start Server-Sent Events for real-time progress
    const eventSource = new EventSource('http://127.0.0.1:8000/train/progress');
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const status = JSON.parse(event.data);
        setTrainingStatus(status);
        
        if (status.status === 'completed') {
          setIsTraining(false);
          addLog('✅ Training completed successfully!');
          onComplete();
          eventSource.close();
        } else if (status.status === 'error') {
          setIsTraining(false);
          setError(status.error || 'Training failed');
          addLog(`❌ Training failed: ${status.error}`);
          eventSource.close();
        }
      } catch (err) {
        console.error('Failed to parse progress data:', err);
      }
    };

    eventSource.onerror = () => {
      addLog('⚠️ Lost connection to training progress');
      eventSource.close();
      
      // Fallback to polling
      setTimeout(() => {
        if (isTraining) {
          checkTrainingStatus();
        }
      }, 5000);
    };
  };

  const stopTraining = async () => {
    try {
      // In a real implementation, you'd have a stop endpoint
      // For now, we'll just stop monitoring
      setIsTraining(false);
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      addLog('🛑 Training stopped by user');
    } catch (err) {
      console.error('Failed to stop training:', err);
    }
  };

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, `[${timestamp}] ${message}`]);
  };

  const getProgressColor = (progress: number) => {
    if (progress < 25) return 'bg-red-500';
    if (progress < 50) return 'bg-yellow-500';
    if (progress < 75) return 'bg-blue-500';
    return 'bg-green-500';
  };

  const getMemoryStatus = (usage: number) => {
    if (usage > 7) return { color: 'text-red-500', status: 'Critical' };
    if (usage > 6) return { color: 'text-yellow-500', status: 'High' };
    return { color: 'text-green-500', status: 'Normal' };
  };

  const canStartTraining = appState.totalDuration >= 8;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="bg-card rounded-lg border p-6">
        <h2 className="text-xl font-semibold mb-2">Step 3: Train AI Model</h2>
        <p className="text-muted-foreground">
          Train your personalized voice model using advanced AI technology.
        </p>
      </div>

      {/* Training Prerequisites */}
      <div className="bg-card rounded-lg border p-6">
        <h3 className="text-lg font-medium mb-4">Training Data Summary</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-blue-600">
              {appState.clipCount}
            </div>
            <div className="text-sm text-muted-foreground">Voice Clips</div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-green-600">
              {appState.totalDuration.toFixed(1)}m
            </div>
            <div className="text-sm text-muted-foreground">Total Audio</div>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-950/20 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-purple-600">
              {appState.avgQuality.toFixed(0)}%
            </div>
            <div className="text-sm text-muted-foreground">Avg Quality</div>
          </div>
        </div>

        {!canStartTraining && (
          <div className="flex items-center gap-2 text-yellow-600 bg-yellow-50 dark:bg-yellow-950/20 p-3 rounded-lg mb-4">
            <AlertTriangle className="h-4 w-4" />
            Need at least 8 minutes of audio for training. Please record more clips.
          </div>
        )}
      </div>

      {/* Memory Monitor */}
      <div className="bg-card rounded-lg border p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MemoryStick className="h-4 w-4" />
            <span className="text-sm font-medium">Memory Usage</span>
          </div>
          <div className="flex items-center gap-2">
            <span className={cn("text-sm font-medium", getMemoryStatus(memoryUsage).color)}>
              {memoryUsage.toFixed(1)} GB
            </span>
            <span className={cn("px-2 py-1 rounded-full text-xs", 
              memoryUsage > 7 ? "bg-red-100 text-red-800" :
              memoryUsage > 6 ? "bg-yellow-100 text-yellow-800" :
              "bg-green-100 text-green-800"
            )}>
              {getMemoryStatus(memoryUsage).status}
            </span>
          </div>
        </div>
        
        <div className="w-full bg-muted rounded-full h-2 mt-2">
          <div 
            className={cn("h-2 rounded-full transition-all",
              memoryUsage > 7 ? "bg-red-500" :
              memoryUsage > 6 ? "bg-yellow-500" :
              "bg-green-500"
            )}
            style={{ width: `${Math.min((memoryUsage / 8) * 100, 100)}%` }}
          />
        </div>
      </div>

      {/* Training Controls */}
      <div className="bg-card rounded-lg border p-6">
        {!isTraining && !trainingStatus?.status === 'completed' ? (
          <div className="text-center space-y-4">
            <div className="space-y-2">
              <h3 className="text-lg font-medium">Ready to Train</h3>
              <p className="text-muted-foreground">
                This will create your personalized AI voice model using advanced deep learning.
              </p>
            </div>
            
            <button
              onClick={startTraining}
              disabled={!canStartTraining}
              className={cn(
                "flex items-center gap-2 mx-auto px-6 py-3 rounded-lg font-medium",
                !canStartTraining 
                  ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                  : "bg-blue-500 hover:bg-blue-600 text-white"
              )}
            >
              <Brain className="h-5 w-5" />
              Start AI Training
            </button>
          </div>
        ) : trainingStatus?.status === 'completed' ? (
          <div className="text-center space-y-4">
            <div className="flex items-center justify-center gap-2 text-green-600">
              <CheckCircle className="h-6 w-6" />
              <h3 className="text-lg font-medium">Training Complete!</h3>
            </div>
            <p className="text-muted-foreground">
              Your AI voice model is ready. You can now generate speech with your cloned voice.
            </p>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Progress Header */}
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-medium flex items-center gap-2">
                <Zap className="h-5 w-5 text-blue-500" />
                Training in Progress
              </h3>
              
              <button
                onClick={stopTraining}
                className="flex items-center gap-2 text-red-600 hover:text-red-700 px-3 py-1 rounded-lg border border-red-200 hover:border-red-300"
              >
                <StopCircle className="h-4 w-4" />
                Stop
              </button>
            </div>

            {/* Progress Bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Progress</span>
                <span>{trainingStatus?.progress_pct || 0}%</span>
              </div>
              <div className="w-full bg-muted rounded-full h-3">
                <div 
                  className={cn("h-3 rounded-full transition-all duration-500", 
                    getProgressColor(trainingStatus?.progress_pct || 0)
                  )}
                  style={{ width: `${trainingStatus?.progress_pct || 0}%` }}
                />
              </div>
            </div>

            {/* Training Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-muted/30 rounded-lg p-3 text-center">
                <div className="text-lg font-semibold">{trainingStatus?.current_epoch || 0}</div>
                <div className="text-xs text-muted-foreground">Current Epoch</div>
              </div>
              
              <div className="bg-muted/30 rounded-lg p-3 text-center">
                <div className="text-lg font-semibold flex items-center justify-center gap-1">
                  <Clock className="h-4 w-4" />
                  {trainingStatus?.eta_min || 0}m
                </div>
                <div className="text-xs text-muted-foreground">ETA</div>
              </div>
              
              <div className="bg-muted/30 rounded-lg p-3 text-center">
                <div className="text-lg font-semibold">{trainingStatus?.mode || 'N/A'}</div>
                <div className="text-xs text-muted-foreground">Mode</div>
              </div>
              
              <div className="bg-muted/30 rounded-lg p-3 text-center">
                <div className={cn("text-lg font-semibold", getMemoryStatus(memoryUsage).color)}>
                  {memoryUsage.toFixed(1)}GB
                </div>
                <div className="text-xs text-muted-foreground">Memory</div>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 text-red-600 bg-red-50 dark:bg-red-950/20 p-3 rounded-lg mt-4">
            <AlertTriangle className="h-4 w-4" />
            {error}
          </div>
        )}
      </div>

      {/* Training Logs */}
      {logs.length > 0 && (
        <div className="bg-card rounded-lg border p-4">
          <h4 className="text-sm font-medium mb-3">Training Logs</h4>
          <div className="bg-black rounded-lg p-3 font-mono text-sm text-green-400 max-h-40 overflow-y-auto">
            {logs.map((log, index) => (
              <div key={index} className="mb-1">
                {log}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
