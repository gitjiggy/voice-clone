import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Play, Pause, RotateCcw, Upload, AlertCircle, CheckCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { QualityGate } from './QualityGate';

interface RecordPageProps {
  appState: {
    recordedClips: number;
    targetClips: number;
  };
  onComplete: (data: any) => void;
  onProgress: (data: any) => void;
}

interface RecordingPrompt {
  id: number;
  text: string;
}

export function RecordPage({ appState, onComplete, onProgress }: RecordPageProps) {
  const [prompts, setPrompts] = useState<RecordingPrompt[]>([]);
  const [currentPromptIndex, setCurrentPromptIndex] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioData, setAudioData] = useState<Blob | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState('');
  const [clips, setClips] = useState<any[]>([]);
  const [totalDuration, setTotalDuration] = useState(0);
  const [avgQuality, setAvgQuality] = useState(0);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationRef = useRef<number>();
  const timerRef = useRef<NodeJS.Timeout>();
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Load recording prompts
  useEffect(() => {
    fetchRecordingPrompts();
    fetchExistingClips();
  }, []);

  // Timer for recording
  useEffect(() => {
    if (isRecording) {
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => {
          const newTime = prev + 0.1;
          if (newTime >= 45) {
            stopRecording();
            return 45;
          }
          return newTime;
        });
      }, 100);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRecording]);

  const fetchRecordingPrompts = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/recording-script');
      const data = await response.json();
      const promptsWithIds = data.prompts.map((text: string, index: number) => ({
        id: index + 1,
        text
      }));
      setPrompts(promptsWithIds);
    } catch (error) {
      console.error('Failed to fetch prompts:', error);
    }
  };

  const fetchExistingClips = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/clips');
      const data = await response.json();
      const clipsData = data.clips || [];
      setClips(clipsData);
      
      // Calculate totals
      const total = clipsData.reduce((sum: number, clip: any) => sum + (clip.duration || 0), 0);
      const avgQual = clipsData.length > 0 
        ? clipsData.reduce((sum: number, clip: any) => sum + (clip.quality || 0), 0) / clipsData.length 
        : 0;
      
      setTotalDuration(total / 60); // Convert to minutes
      setAvgQuality(avgQual);
      onProgress({ 
        recordedClips: clipsData.length,
        totalDuration: total / 60, // Pass total duration to dashboard
        avgQuality: avgQual // Pass average quality to dashboard
      });
    } catch (error) {
      console.error('Failed to fetch clips:', error);
    }
  };

  const setupAudioContext = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;
      
      return stream;
    } catch (error) {
      console.error('Failed to setup audio:', error);
      throw error;
    }
  };

  const drawWaveform = () => {
    if (!canvasRef.current || !analyserRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const analyser = analyserRef.current;
    
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(dataArray);
    
    ctx!.clearRect(0, 0, canvas.width, canvas.height);
    
    const barWidth = (canvas.width / dataArray.length) * 2.5;
    let barHeight;
    let x = 0;
    
    const gradient = ctx!.createLinearGradient(0, 0, 0, canvas.height);
    gradient.addColorStop(0, isRecording ? '#ef4444' : '#3b82f6');
    gradient.addColorStop(1, isRecording ? '#dc2626' : '#1d4ed8');
    
    for (let i = 0; i < dataArray.length; i++) {
      barHeight = (dataArray[i] / 255) * canvas.height * 0.8;
      
      ctx!.fillStyle = gradient;
      ctx!.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
      
      x += barWidth + 1;
    }
    
    if (isRecording) {
      animationRef.current = requestAnimationFrame(drawWaveform);
    }
  };

  const startRecording = async () => {
    try {
      setRecordingTime(0);
      setAudioData(null);
      setErrorMessage('');
      
      const stream = await setupAudioContext();
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      
      const chunks: BlobPart[] = [];
      
      mediaRecorder.ondataavailable = (event) => {
        chunks.push(event.data);
      };
      
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        setAudioData(blob);
        
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
        
        // Stop animation
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
        }
      };
      
      mediaRecorder.start();
      setIsRecording(true);
      drawWaveform();
      
    } catch (error) {
      console.error('Recording failed:', error);
      setErrorMessage('Failed to start recording. Please check microphone permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      // Validate duration
      if (recordingTime < 15) {
        setErrorMessage('Recording too short! Please record for 15-30 seconds.');
        setAudioData(null);
      } else if (recordingTime > 30) {
        setErrorMessage('Recording too long! Please keep it between 15-30 seconds.');
        setAudioData(null);
      }
    }
  };

  const playRecording = () => {
    if (audioData && audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
        setIsPlaying(false);
      } else {
        const url = URL.createObjectURL(audioData);
        audioRef.current.src = url;
        audioRef.current.play();
        setIsPlaying(true);
        
        audioRef.current.onended = () => {
          setIsPlaying(false);
          URL.revokeObjectURL(url);
        };
      }
    }
  };

  const uploadRecording = async () => {
    if (!audioData) return;
    
    setUploadStatus('uploading');
    
    try {
      const formData = new FormData();
      formData.append('file', audioData, `recording_${Date.now()}.wav`);
      
      const response = await fetch('http://127.0.0.1:8000/upload', {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      
      if (response.ok) {
        setUploadStatus('success');
        setAudioData(null);
        setRecordingTime(0);
        
        // Update clip count
        const newClipCount = result.total_clips;
        onProgress({ recordedClips: newClipCount });
        
        // Refresh clips data to update quality gate
        await fetchExistingClips();
        
        // Auto-advance to next prompt
        if (currentPromptIndex < prompts.length - 1) {
          setCurrentPromptIndex(prev => prev + 1);
        }
        
        // Allow user to continue recording even after reaching target
        // Only mark as complete if analysis shows sufficient audio quality
        if (newClipCount >= appState.targetClips) {
          // Mark step as ready but don't force completion - let user record more if needed
          onComplete({ recordedClips: newClipCount, targetClips: appState.targetClips });
        }
        
        // Reset status after 2 seconds
        setTimeout(() => setUploadStatus('idle'), 2000);
        
      } else {
        setUploadStatus('error');
        setErrorMessage(result.detail || 'Upload failed');
        setTimeout(() => setUploadStatus('idle'), 3000);
      }
      
    } catch (error) {
      setUploadStatus('error');
      setErrorMessage('Network error during upload');
      setTimeout(() => setUploadStatus('idle'), 3000);
    }
  };

  const resetRecording = () => {
    setAudioData(null);
    setRecordingTime(0);
    setErrorMessage('');
    setUploadStatus('idle');
  };

  const getTimerColor = () => {
    if (recordingTime < 15) return 'text-yellow-500';
    if (recordingTime <= 30) return 'text-green-500';
    return 'text-red-500';
  };

  const currentPrompt = prompts[currentPromptIndex];
  const progress = (appState.recordedClips / appState.targetClips) * 100;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Progress Header */}
      <div className="bg-card rounded-lg border p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Step 1: Record Your Voice</h2>
          <div className="text-sm text-muted-foreground">
            {appState.recordedClips} of {appState.targetClips} clips recorded
          </div>
        </div>
        
        <div className="w-full bg-muted rounded-full h-2 mb-2">
          <div 
            className="bg-blue-500 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="text-xs text-muted-foreground">
          {Math.min(progress, 100).toFixed(0)}% complete
        </div>
        
        {appState.recordedClips >= appState.targetClips && (
          <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800 rounded-lg">
            <p className="text-sm text-blue-800 dark:text-blue-200">
              🎯 <strong>Target reached!</strong> You can record more clips for better quality, or click <strong>"2"</strong> above to analyze your recordings.
            </p>
          </div>
        )}
      </div>

      {/* Quality Gate */}
      <QualityGate 
        totalDuration={totalDuration}
        clipCount={appState.recordedClips}
        avgQuality={avgQuality}
        targetClips={appState.targetClips}
      />

      {/* Current Prompt */}
      <div className="bg-card rounded-lg border p-6">
        <h3 className="text-lg font-medium mb-4">
          Phrase {currentPromptIndex + 1} of {prompts.length}
        </h3>
        
        {currentPrompt && (
          <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-4 mb-4">
            <p className="text-lg leading-relaxed">
              "{currentPrompt.text}"
            </p>
          </div>
        )}
        
        <div className="text-sm text-muted-foreground mb-4">
          <strong>Recording guidance:</strong> Record for 15-30 seconds - speak naturally and clearly.
        </div>
      </div>

      {/* Recording Interface */}
      <div className="bg-card rounded-lg border p-6">
        {/* Waveform Visualization */}
        <div className="mb-6">
          <canvas
            ref={canvasRef}
            width={600}
            height={100}
            className="w-full h-20 bg-muted/30 rounded-lg"
          />
        </div>

        {/* Timer */}
        <div className="text-center mb-6">
          <div className={cn("text-3xl font-mono font-bold", getTimerColor())}>
            {recordingTime.toFixed(1)}s
          </div>
          <div className="text-sm text-muted-foreground">
            {recordingTime < 15 ? 'Keep recording...' : 
             recordingTime <= 30 ? 'Good duration!' : 
             'Too long - please re-record'}
          </div>
        </div>

        {/* Recording Controls */}
        <div className="flex justify-center gap-4 mb-6">
          {!isRecording ? (
            <button
              onClick={startRecording}
              className="flex items-center gap-2 bg-red-500 hover:bg-red-600 text-white px-6 py-3 rounded-lg font-medium"
            >
              <Mic className="h-5 w-5" />
              Start Recording
            </button>
          ) : (
            <button
              onClick={stopRecording}
              className="flex items-center gap-2 bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg font-medium animate-pulse"
            >
              <MicOff className="h-5 w-5" />
              Stop Recording
            </button>
          )}

          {audioData && (
            <>
              <button
                onClick={playRecording}
                className="flex items-center gap-2 bg-blue-500 hover:bg-blue-600 text-white px-4 py-3 rounded-lg"
              >
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                {isPlaying ? 'Pause' : 'Play'}
              </button>

              <button
                onClick={resetRecording}
                className="flex items-center gap-2 bg-gray-500 hover:bg-gray-600 text-white px-4 py-3 rounded-lg"
              >
                <RotateCcw className="h-4 w-4" />
                Reset
              </button>

              <button
                onClick={uploadRecording}
                disabled={uploadStatus === 'uploading' || recordingTime < 15 || recordingTime > 30}
                className={cn(
                  "flex items-center gap-2 px-4 py-3 rounded-lg font-medium",
                  uploadStatus === 'success' ? "bg-green-500 text-white" :
                  uploadStatus === 'error' ? "bg-red-500 text-white" :
                  uploadStatus === 'uploading' ? "bg-blue-400 text-white" :
                  (recordingTime < 30 || recordingTime > 45) ? "bg-gray-300 text-gray-500 cursor-not-allowed" :
                  "bg-green-500 hover:bg-green-600 text-white"
                )}
              >
                {uploadStatus === 'uploading' ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Uploading...
                  </>
                ) : uploadStatus === 'success' ? (
                  <>
                    <CheckCircle className="h-4 w-4" />
                    Uploaded!
                  </>
                ) : (
                  <>
                    <Upload className="h-4 w-4" />
                    Upload Clip
                  </>
                )}
              </button>
            </>
          )}
        </div>

        {/* Error Messages */}
        {errorMessage && (
          <div className="flex items-center gap-2 text-red-600 bg-red-50 dark:bg-red-950/20 p-3 rounded-lg">
            <AlertCircle className="h-4 w-4" />
            {errorMessage}
          </div>
        )}

        {/* Hidden audio element */}
        <audio ref={audioRef} style={{ display: 'none' }} />
      </div>

      {/* Navigation Controls */}
      {currentPromptIndex < prompts.length - 1 && (
        <div className="text-center">
          <button
            onClick={() => setCurrentPromptIndex(prev => prev + 1)}
            className="text-blue-600 hover:text-blue-700 font-medium"
          >
            Skip to next phrase →
          </button>
        </div>
      )}
    </div>
  );
}
