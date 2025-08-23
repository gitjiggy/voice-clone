import React, { useState } from 'react';
import { Brain, CheckCircle, AlertCircle, BarChart3, Clock, Star } from 'lucide-react';
import { cn } from '@/lib/utils';

interface AnalyzePageProps {
  appState: {
    recordedClips: number;
    targetClips: number;
  };
  onComplete: (data: any) => void;
}

interface AnalysisResult {
  total_duration_min: number;
  clip_count: number;
  avg_quality: number;
  recommended_mode: string;
}

export function AnalyzePage({ appState, onComplete }: AnalyzePageProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState('');

  const startAnalysis = async () => {
    setIsAnalyzing(true);
    setError('');
    
    try {
      const response = await fetch('http://127.0.0.1:8000/analyze-dataset', {
        method: 'POST'
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setAnalysisResult(data);
      } else {
        setError(data.detail || 'Analysis failed');
      }
    } catch (err) {
      setError('Network error during analysis');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const continueToTraining = () => {
    if (analysisResult) {
      onComplete({
        totalDuration: analysisResult.total_duration_min,
        avgQuality: analysisResult.avg_quality,
        clipCount: analysisResult.clip_count,
        recommendedMode: analysisResult.recommended_mode
      });
    }
  };

  const getQualityColor = (quality: number) => {
    if (quality >= 80) return 'text-green-600';
    if (quality >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getQualityBadge = (quality: number) => {
    if (quality >= 80) return { text: 'Excellent', color: 'bg-green-100 text-green-800' };
    if (quality >= 60) return { text: 'Good', color: 'bg-yellow-100 text-yellow-800' };
    return { text: 'Fair', color: 'bg-red-100 text-red-800' };
  };

  const getDurationStatus = (duration: number) => {
    if (duration < 8) return {
      status: 'need_more',
      color: 'text-yellow-600',
      message: 'Need more recordings for optimal training'
    };
    if (duration < 12) return {
      status: 'quick_training',
      color: 'text-blue-600',
      message: 'Ready for quick training (15-30 min)'
    };
    return {
      status: 'full_training',
      color: 'text-green-600',
      message: 'Ready for full training (30-45 min)'
    };
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="bg-card rounded-lg border p-6">
        <h2 className="text-xl font-semibold mb-2">Step 2: Analyze Voice Data</h2>
        <p className="text-muted-foreground">
          Let's analyze your recorded voice data to prepare for AI training.
        </p>
      </div>

      {/* Analysis Status */}
      <div className="bg-card rounded-lg border p-6">
        <h3 className="text-lg font-medium mb-4">Dataset Overview</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-blue-600">
              {appState.recordedClips}
            </div>
            <div className="text-sm text-muted-foreground">Clips Recorded</div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-green-600">
              {appState.targetClips}
            </div>
            <div className="text-sm text-muted-foreground">Target Clips</div>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-950/20 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-purple-600">
              {((appState.recordedClips / appState.targetClips) * 100).toFixed(0)}%
            </div>
            <div className="text-sm text-muted-foreground">Completion</div>
          </div>
        </div>

        {!analysisResult ? (
          <div className="text-center">
            <button
              onClick={startAnalysis}
              disabled={isAnalyzing || appState.recordedClips === 0}
              className={cn(
                "flex items-center gap-2 mx-auto px-6 py-3 rounded-lg font-medium",
                isAnalyzing 
                  ? "bg-blue-400 text-white cursor-not-allowed"
                  : appState.recordedClips === 0
                    ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                    : "bg-blue-500 hover:bg-blue-600 text-white"
              )}
            >
              {isAnalyzing ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Analyzing Voice Data...
                </>
              ) : (
                <>
                  <Brain className="h-5 w-5" />
                  Analyze Voice Data
                </>
              )}
            </button>
            
            {appState.recordedClips === 0 && (
              <p className="text-sm text-muted-foreground mt-2">
                Please record some clips first
              </p>
            )}
          </div>
        ) : null}

        {error && (
          <div className="flex items-center gap-2 text-red-600 bg-red-50 dark:bg-red-950/20 p-3 rounded-lg">
            <AlertCircle className="h-4 w-4" />
            {error}
          </div>
        )}
      </div>

      {/* Analysis Results */}
      {analysisResult && (
        <div className="bg-card rounded-lg border p-6">
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle className="h-5 w-5 text-green-500" />
            <h3 className="text-lg font-medium">Analysis Complete!</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            {/* Duration Analysis */}
            <div className="space-y-4">
              <h4 className="font-medium flex items-center gap-2">
                <Clock className="h-4 w-4" />
                Duration Analysis
              </h4>
              
              <div className="bg-muted/30 rounded-lg p-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-muted-foreground">Total Audio</span>
                  <span className="font-medium">{analysisResult.total_duration_min.toFixed(1)} minutes</span>
                </div>
                
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-muted-foreground">Clips Count</span>
                  <span className="font-medium">{analysisResult.clip_count}</span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Status</span>
                  <span className={cn("font-medium", getDurationStatus(analysisResult.total_duration_min).color)}>
                    {getDurationStatus(analysisResult.total_duration_min).status.replace('_', ' ')}
                  </span>
                </div>
              </div>
              
              <div className={cn("text-sm p-3 rounded-lg", 
                getDurationStatus(analysisResult.total_duration_min).color === 'text-green-600' 
                  ? 'bg-green-50 dark:bg-green-950/20 text-green-700'
                  : getDurationStatus(analysisResult.total_duration_min).color === 'text-blue-600'
                    ? 'bg-blue-50 dark:bg-blue-950/20 text-blue-700'
                    : 'bg-yellow-50 dark:bg-yellow-950/20 text-yellow-700'
              )}>
                {getDurationStatus(analysisResult.total_duration_min).message}
              </div>
            </div>

            {/* Quality Analysis */}
            <div className="space-y-4">
              <h4 className="font-medium flex items-center gap-2">
                <Star className="h-4 w-4" />
                Quality Analysis
              </h4>
              
              <div className="bg-muted/30 rounded-lg p-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-muted-foreground">Average Quality</span>
                  <span className={cn("font-medium", getQualityColor(analysisResult.avg_quality))}>
                    {analysisResult.avg_quality.toFixed(1)}%
                  </span>
                </div>
                
                <div className="w-full bg-muted rounded-full h-2 mb-3">
                  <div 
                    className={cn("h-2 rounded-full transition-all",
                      analysisResult.avg_quality >= 80 ? "bg-green-500" :
                      analysisResult.avg_quality >= 60 ? "bg-yellow-500" :
                      "bg-red-500"
                    )}
                    style={{ width: `${analysisResult.avg_quality}%` }}
                  />
                </div>
                
                <div className="flex justify-center">
                  <span className={cn("px-2 py-1 rounded-full text-xs font-medium",
                    getQualityBadge(analysisResult.avg_quality).color
                  )}>
                    {getQualityBadge(analysisResult.avg_quality).text}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Training Recommendation */}
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 rounded-lg p-6">
            <h4 className="font-medium mb-3 flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Training Recommendation
            </h4>
            
            <div className="text-center">
              <div className="mb-4">
                <div className="text-lg font-semibold text-blue-600 mb-2">
                  Continue with AI Training
                </div>
                <div className="text-sm text-muted-foreground">
                  Your voice data is ready for AI model training. This is the only path forward to create your personalized voice clone.
                </div>
              </div>

              <button
                onClick={continueToTraining}
                className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-lg font-medium inline-flex items-center gap-2"
              >
                <Brain className="h-4 w-4" />
                Continue with AI Training
              </button>
            </div>
          </div>

          {/* Technical Details */}
          <details className="mt-4">
            <summary className="cursor-pointer text-sm text-muted-foreground hover:text-foreground">
              View Technical Details
            </summary>
            <div className="mt-3 p-4 bg-muted/30 rounded-lg text-sm space-y-2">
              <div><strong>Recommended Mode:</strong> {analysisResult.recommended_mode}</div>
              <div><strong>Expected Training Time:</strong> {
                analysisResult.total_duration_min < 8 ? 'N/A (need more data)' :
                analysisResult.total_duration_min < 12 ? '15-30 minutes' :
                '30-45 minutes'
              }</div>
              <div><strong>Model Type:</strong> XTTS v2 (Real Voice Clone)</div>
            </div>
          </details>
        </div>
      )}
    </div>
  );
}
