import React from 'react';
import { CheckCircle, AlertCircle, XCircle, Clock } from 'lucide-react';
import { cn } from '@/lib/utils';

interface QualityGateProps {
  totalDuration: number; // in minutes
  clipCount: number;
  avgQuality: number;
  targetClips: number;
}

type QualityStatus = 'green' | 'yellow' | 'red';

export function QualityGate({ totalDuration, clipCount, avgQuality, targetClips }: QualityGateProps) {
  
  const getQualityStatus = (): QualityStatus => {
    if (totalDuration >= 8 && avgQuality >= 60) return 'green';
    if (totalDuration >= 5) return 'yellow';
    return 'red';
  };

  const status = getQualityStatus();
  
  const statusConfig = {
    green: {
      icon: CheckCircle,
      title: "Ready for Training",
      subtitle: "High quality audio dataset ready",
      bgColor: "bg-green-50 dark:bg-green-950/20",
      borderColor: "border-green-200 dark:border-green-800",
      textColor: "text-green-800 dark:text-green-200",
      iconColor: "text-green-600",
      message: `Excellent! You have ${totalDuration.toFixed(1)} minutes of high-quality audio. Your voice clone will be very accurate.`
    },
    yellow: {
      icon: AlertCircle,
      title: "Need More Audio",
      subtitle: "Continue recording to reach optimal quality",
      bgColor: "bg-yellow-50 dark:bg-yellow-950/20",
      borderColor: "border-yellow-200 dark:border-yellow-800",
      textColor: "text-yellow-800 dark:text-yellow-200",
      iconColor: "text-yellow-600",
      message: `You have ${totalDuration.toFixed(1)} minutes. Record ${Math.max(0, 20 - clipCount)} more clips to reach 8+ minutes for optimal training.`
    },
    red: {
      icon: XCircle,
      title: "Insufficient Audio",
      subtitle: "Must record more clips before training",
      bgColor: "bg-red-50 dark:bg-red-950/20",
      borderColor: "border-red-200 dark:border-red-800",
      textColor: "text-red-800 dark:text-red-200",
      iconColor: "text-red-600",
      message: `Only ${totalDuration.toFixed(1)} minutes recorded. You need at least 5 minutes (minimum) to start training. Record ${Math.max(0, 20 - clipCount)} more clips.`
    }
  };

  const config = statusConfig[status];
  const Icon = config.icon;

  const getProgressPercentage = () => {
    const minRequired = 5; // minimum 5 minutes
    const optimal = 10; // optimal 10 minutes (20 clips × 30s)
    
    if (totalDuration >= optimal) return 100;
    return Math.min((totalDuration / optimal) * 100, 100);
  };

  const getProgressColor = () => {
    if (status === 'green') return 'bg-green-500';
    if (status === 'yellow') return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className={cn("rounded-lg border p-6", config.bgColor, config.borderColor)}>
      <div className="flex items-start gap-4">
        <Icon className={cn("h-6 w-6 mt-1", config.iconColor)} />
        
        <div className="flex-1">
          <div className="flex items-center justify-between mb-2">
            <h3 className={cn("text-lg font-semibold", config.textColor)}>
              {config.title}
            </h3>
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4" />
              <span className="text-sm font-medium">
                {totalDuration.toFixed(1)}min / 10min optimal
              </span>
            </div>
          </div>
          
          <p className={cn("text-sm mb-4", config.textColor)}>
            {config.subtitle}
          </p>
          
          {/* Progress Bar */}
          <div className="mb-4">
            <div className="flex justify-between text-xs mb-1">
              <span>Audio Quality Progress</span>
              <span>{getProgressPercentage().toFixed(0)}%</span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div 
                className={cn("h-2 rounded-full transition-all duration-500", getProgressColor())}
                style={{ width: `${getProgressPercentage()}%` }}
              />
            </div>
            <div className="flex justify-between text-xs mt-1 text-muted-foreground">
              <span>5min (minimum)</span>
              <span>8min (good)</span>
              <span>10min (optimal)</span>
            </div>
          </div>
          
          {/* Detailed Stats */}
          <div className="grid grid-cols-3 gap-4 mb-4 text-center">
            <div className="bg-white/50 dark:bg-black/20 rounded-lg p-3">
              <div className="text-lg font-bold">{clipCount}</div>
              <div className="text-xs text-muted-foreground">Clips</div>
            </div>
            <div className="bg-white/50 dark:bg-black/20 rounded-lg p-3">
              <div className="text-lg font-bold">{avgQuality.toFixed(0)}%</div>
              <div className="text-xs text-muted-foreground">Quality</div>
            </div>
            <div className="bg-white/50 dark:bg-black/20 rounded-lg p-3">
              <div className="text-lg font-bold">{(totalDuration / clipCount || 0).toFixed(1)}s</div>
              <div className="text-xs text-muted-foreground">Avg Length</div>
            </div>
          </div>
          
          {/* Status Message */}
          <div className={cn("text-sm font-medium", config.textColor)}>
            {config.message}
          </div>
          
          {/* Quality Commitment */}
          {status === 'green' && (
            <div className="mt-4 p-3 bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-950/30 dark:to-green-950/30 rounded-lg border border-blue-200 dark:border-blue-800">
              <div className="flex items-center gap-2 text-blue-800 dark:text-blue-200">
                <CheckCircle className="h-4 w-4" />
                <span className="font-medium">Quality Commitment:</span>
              </div>
              <p className="text-sm text-blue-700 dark:text-blue-300 mt-1">
                Your actual voice will be cloned - not approximations or synthetic audio. 
                The system will generate speech that sounds like YOU.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
