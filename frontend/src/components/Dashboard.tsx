import React, { useState, useEffect } from 'react';
import { CheckCircle, Circle, ArrowRight, Trash2, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { RecordPage } from './RecordPage';
import { AnalyzePage } from './AnalyzePage';
import { TrainPage } from './TrainPage';
import { DemoPage } from './DemoPage';

interface Step {
  id: number;
  title: string;
  description: string;
  completed: boolean;
  active: boolean;
}

export function Dashboard() {
  const [currentStep, setCurrentStep] = useState(1);
  const [steps, setSteps] = useState<Step[]>([
    {
      id: 1,
      title: "Record",
      description: "Record 20 voice clips",
      completed: false,
      active: true
    },
    {
      id: 2,
      title: "Analyze",
      description: "Analyze voice data",
      completed: false,
      active: false
    },
    {
      id: 3,
      title: "Train",
      description: "Train AI model",
      completed: false,
      active: false
    },
    {
      id: 4,
      title: "Demo",
      description: "Test your voice",
      completed: false,
      active: false
    }
  ]);

  const [appState, setAppState] = useState({
    recordedClips: 0,
    targetClips: 20,
    datasetAnalyzed: false,
    modelTrained: false,
    totalDuration: 0,
    avgQuality: 0
  });
  
  const [showResetDialog, setShowResetDialog] = useState(false);
  const [isResetting, setIsResetting] = useState(false);

  // Load saved state from localStorage
  useEffect(() => {
    const savedState = localStorage.getItem('voice-clone-state');
    if (savedState) {
      try {
        const parsed = JSON.parse(savedState);
        setAppState(parsed);
        
        // Update steps based on saved state - but allow returning to record more clips
        if (parsed.recordedClips >= parsed.targetClips) {
          updateStepCompletion(1, true);
          // Only auto-advance if analysis is also complete AND we have enough audio
          if (!steps[1].active && parsed.datasetAnalyzed && parsed.totalDuration >= 8) {
            setCurrentStep(2);
          }
        }
        if (parsed.datasetAnalyzed) {
          updateStepCompletion(2, true);
          if (!steps[2].active && parsed.recordedClips >= parsed.targetClips) setCurrentStep(3);
        }
        if (parsed.modelTrained) {
          updateStepCompletion(3, true);
          if (!steps[3].active) setCurrentStep(4);
        }
      } catch (e) {
        console.error('Failed to load saved state:', e);
      }
    }
  }, []);

  // Save state to localStorage
  useEffect(() => {
    localStorage.setItem('voice-clone-state', JSON.stringify(appState));
  }, [appState]);

  const updateStepCompletion = (stepId: number, completed: boolean) => {
    setSteps(prev => prev.map(step => 
      step.id === stepId ? { ...step, completed } : step
    ));
  };

  const advanceToStep = (stepId: number) => {
    setCurrentStep(stepId);
    setSteps(prev => prev.map(step => ({
      ...step,
      active: step.id === stepId
    })));
  };

  const handleStepComplete = (stepId: number, data?: any) => {
    updateStepCompletion(stepId, true);
    
    if (stepId === 1 && data?.recordedClips >= data?.targetClips) {
      setAppState(prev => ({ ...prev, recordedClips: data.recordedClips }));
      advanceToStep(2);
    } else if (stepId === 2) {
      setAppState(prev => ({ 
        ...prev, 
        datasetAnalyzed: true,
        totalDuration: data?.totalDuration || 0,
        avgQuality: data?.avgQuality || 0
      }));
      advanceToStep(3);
    } else if (stepId === 3) {
      setAppState(prev => ({ ...prev, modelTrained: true }));
      advanceToStep(4);
    }
  };

  const handleReset = async () => {
    setIsResetting(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/reset-training-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      const result = await response.json();
      
      if (response.ok && result.status === 'success') {
        // Reset all state
        setAppState({
          recordedClips: 0,
          targetClips: 20,
          datasetAnalyzed: false,
          modelTrained: false,
          totalDuration: 0,
          avgQuality: 0
        });
        
        // Reset steps
        setSteps([
          { id: 1, title: "Record", description: "Record 20 voice clips", completed: false, active: true },
          { id: 2, title: "Analyze", description: "Process audio data", completed: false, active: false },
          { id: 3, title: "Train", description: "Build voice model", completed: false, active: false },
          { id: 4, title: "Demo", description: "Test your voice", completed: false, active: false }
        ]);
        
        setCurrentStep(1);
        
        // Clear localStorage
        localStorage.removeItem('voice-clone-state');
        
        setShowResetDialog(false);
        
        // Show success message
        alert(`✅ Clean slate completed!\n\nRemoved ${result.total_files_removed} files:\n• ${result.reset_summary.audio_clips} audio clips\n• ${result.reset_summary.model_files} model files\n• ${result.reset_summary.output_files} output files\n\nReady for fresh training!`);
        
      } else {
        throw new Error(result.error || 'Reset failed');
      }
    } catch (error) {
      console.error('Reset error:', error);
      alert(`❌ Reset failed: ${error.message}`);
    } finally {
      setIsResetting(false);
    }
  };

  const renderProgressTracker = () => (
    <div className="bg-card rounded-lg border p-6 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Voice Clone Progress</h2>
        
        {/* Reset Button */}
        <button
          onClick={() => setShowResetDialog(true)}
          className="flex items-center gap-2 px-3 py-2 text-sm text-red-600 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-950/20 rounded-lg transition-colors"
          title="Clear all data and start fresh"
        >
          <Trash2 className="h-4 w-4" />
          <span className="hidden sm:inline">Reset</span>
        </button>
      </div>
      
      <div className="flex items-center justify-between overflow-x-auto pb-2">
        {steps.map((step, index) => (
          <React.Fragment key={step.id}>
            {/* Step Circle */}
            <div className="flex flex-col items-center min-w-0 flex-shrink-0">
              <button
                onClick={() => setCurrentStep(step.id)}
                className={cn(
                  "w-10 h-10 md:w-12 md:h-12 rounded-full border-2 flex items-center justify-center mb-2 transition-all hover:scale-105 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2",
                  step.completed 
                    ? "bg-green-500 border-green-500 text-white hover:bg-green-600" 
                    : step.active 
                      ? "bg-blue-500 border-blue-500 text-white"
                      : "bg-muted border-muted-foreground/20 text-muted-foreground hover:bg-muted/80"
                )}
              >
                {step.completed ? (
                  <CheckCircle className="h-4 w-4 md:h-6 md:w-6" />
                ) : (
                  <span className="font-medium text-sm md:text-base">{step.id}</span>
                )}
              </button>
              
              {/* Step Labels */}
              <div className="text-center">
                <div className={cn(
                  "font-medium text-xs md:text-sm",
                  step.active ? "text-blue-600" : step.completed ? "text-green-600" : "text-muted-foreground"
                )}>
                  {step.title}
                </div>
                <div className="text-xs text-muted-foreground hidden md:block">
                  {step.description}
                </div>
              </div>
            </div>
            
            {/* Arrow between steps */}
            {index < steps.length - 1 && (
              <ArrowRight className={cn(
                "h-4 w-4 md:h-5 md:w-5 mx-2 md:mx-4 flex-shrink-0",
                steps[index + 1].active || steps[index + 1].completed 
                  ? "text-blue-500" 
                  : "text-muted-foreground/30"
              )} />
            )}
          </React.Fragment>
        ))}
      </div>
      
      {/* Progress Summary */}
      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
        <div className="bg-muted/30 rounded-lg p-3">
          <div className="text-lg font-semibold">{appState.recordedClips}</div>
          <div className="text-xs text-muted-foreground">Clips Recorded</div>
        </div>
        <div className="bg-muted/30 rounded-lg p-3">
          <div className="text-lg font-semibold">{appState.totalDuration.toFixed(1)}m</div>
          <div className="text-xs text-muted-foreground">Total Audio</div>
        </div>
        <div className="bg-muted/30 rounded-lg p-3">
          <div className="text-lg font-semibold">{appState.avgQuality.toFixed(0)}%</div>
          <div className="text-xs text-muted-foreground">Avg Quality</div>
        </div>
        <div className="bg-muted/30 rounded-lg p-3">
          <div className="text-lg font-semibold">
            {appState.modelTrained ? "✓" : "○"}
          </div>
          <div className="text-xs text-muted-foreground">Model Ready</div>
        </div>
      </div>
    </div>
  );

  const renderCurrentPage = () => {
    switch (currentStep) {
      case 1:
        return (
          <RecordPage 
            appState={appState}
            onComplete={(data) => handleStepComplete(1, data)}
            onProgress={(data) => setAppState(prev => ({ ...prev, ...data }))}
          />
        );
      case 2:
        return (
          <AnalyzePage 
            appState={appState}
            onComplete={(data) => handleStepComplete(2, data)}
          />
        );
      case 3:
        return (
          <TrainPage 
            appState={appState}
            onComplete={() => handleStepComplete(3)}
          />
        );
      case 4:
        return (
          <DemoPage 
            appState={appState}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold">Voice Clone Hackathon</h1>
          <p className="text-muted-foreground">Create your AI voice in 4 simple steps</p>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        {renderProgressTracker()}
        {renderCurrentPage()}
      </main>

      {/* Reset Confirmation Dialog */}
      {showResetDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-card rounded-lg border max-w-md w-full p-6">
            <div className="flex items-center gap-3 mb-4">
              <AlertTriangle className="h-6 w-6 text-red-500" />
              <h3 className="text-lg font-semibold">Reset All Training Data</h3>
            </div>
            
            <p className="text-muted-foreground mb-6">
              This will permanently delete:
            </p>
            
            <ul className="text-sm space-y-2 mb-6 text-muted-foreground">
              <li>• All recorded audio clips</li>
              <li>• Training dataset and analysis</li>
              <li>• Trained voice models</li>
              <li>• Generated audio outputs</li>
              <li>• All progress and settings</li>
            </ul>
            
            <div className="bg-yellow-50 dark:bg-yellow-950/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-3 mb-6">
              <p className="text-sm text-yellow-800 dark:text-yellow-200">
                ⚠️ This action cannot be undone. You'll start with a completely clean slate.
              </p>
            </div>
            
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowResetDialog(false)}
                className="px-4 py-2 text-sm border rounded-lg hover:bg-muted/50 transition-colors"
                disabled={isResetting}
              >
                Cancel
              </button>
              <button
                onClick={handleReset}
                disabled={isResetting}
                className="px-4 py-2 text-sm bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
              >
                {isResetting ? (
                  <>
                    <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                    Resetting...
                  </>
                ) : (
                  <>
                    <Trash2 className="h-4 w-4" />
                    Reset Everything
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
