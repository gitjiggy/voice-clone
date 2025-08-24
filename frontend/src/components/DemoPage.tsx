import React, { useState, useRef } from 'react';
import { Play, Pause, Download, Volume2, Thermometer, AlertCircle, CheckCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface DemoPageProps {
  appState: {
    modelTrained: boolean;
  };
}

export function DemoPage({ appState }: DemoPageProps) {
  const [text, setText] = useState('');
  const [speed, setSpeed] = useState(1.0);
  const [temperature, setTemperature] = useState(0.7);
  const [isGenerating, setIsGenerating] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const presetPhrases = [
    "Hello, this is my cloned voice!",
    "Welcome to our hackathon demo!",
    "AI voice cloning technology is truly amazing.",
    "Thank you for trying out my personalized voice assistant.",
    "This voice was created using advanced machine learning techniques."
  ];

  const generateSpeech = async () => {
    if (!text.trim()) {
      setError('Please enter some text to generate speech');
      return;
    }

    if (text.length > 1000) {
      setError('Text is too long. Please keep it under 1000 characters.');
      return;
    }

    setIsGenerating(true);
    setError('');
    setSuccess('');

    try {
      const response = await fetch('http://127.0.0.1:8000/speak', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text.trim(),
          speed: speed,
          temperature: temperature
        })
      });

      if (response.ok) {
        const audioBlob = await response.blob();
        const url = URL.createObjectURL(audioBlob);
        setAudioUrl(url);
        setSuccess('Voice generated successfully!');
        
        // Auto-play the generated audio
        if (audioRef.current) {
          audioRef.current.src = url;
          audioRef.current.load();
        }
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to generate speech');
      }
    } catch (err) {
      setError('Network error during speech generation');
    } finally {
      setIsGenerating(false);
    }
  };

  const playAudio = () => {
    if (audioRef.current && audioUrl) {
      if (isPlaying) {
        audioRef.current.pause();
        setIsPlaying(false);
      } else {
        audioRef.current.play();
        setIsPlaying(true);
      }
    }
  };

  const downloadAudio = () => {
    if (audioUrl) {
      const a = document.createElement('a');
      a.href = audioUrl;
      a.download = `voice_clone_${Date.now()}.wav`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  const usePresetPhrase = (phrase: string) => {
    setText(phrase);
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
  };

  const handleAudioError = () => {
    setError('Failed to play audio');
    setIsPlaying(false);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="bg-card rounded-lg border p-6">
        <h2 className="text-xl font-semibold mb-2">Step 4: Demo Your Voice</h2>
        <p className="text-muted-foreground">
          Test your personalized AI voice model with custom text.
        </p>
      </div>

      {/* Voice Mode Indicator */}
      <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-950/20 dark:to-blue-950/20 rounded-lg border p-4">
        <div className="flex items-center justify-center gap-2">
          <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
          <span className="font-medium">Using your trained voice</span>
          <span className="text-sm text-muted-foreground">• Real AI voice clone active</span>
        </div>
      </div>

      {/* Model Status Check */}
      {!appState.modelTrained && (
        <div className="flex items-center gap-2 text-yellow-600 bg-yellow-50 dark:bg-yellow-950/20 p-4 rounded-lg border">
          <AlertCircle className="h-5 w-5" />
          <div>
            <div className="font-medium">Model not ready yet</div>
            <div className="text-sm">Please complete the training step first to use voice synthesis.</div>
          </div>
        </div>
      )}

      {/* Text Input */}
      <div className="bg-card rounded-lg border p-6">
        <h3 className="text-lg font-medium mb-4">Text to Speech</h3>
        
        {/* Preset Phrases */}
        <div className="mb-4">
          <h4 className="text-sm font-medium mb-2">Quick Presets:</h4>
          <div className="flex flex-wrap gap-2">
            {presetPhrases.map((phrase, index) => (
              <button
                key={index}
                onClick={() => usePresetPhrase(phrase)}
                className="px-3 py-1 text-sm bg-blue-100 hover:bg-blue-200 dark:bg-blue-950/30 dark:hover:bg-blue-950/50 text-blue-700 dark:text-blue-300 rounded-full transition-colors"
              >
                "{phrase.substring(0, 30)}{phrase.length > 30 ? '...' : ''}"
              </button>
            ))}
          </div>
        </div>

        {/* Text Input Area */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="text-sm font-medium">Enter your text:</label>
            <span className={cn("text-xs", 
              text.length > 1000 ? "text-red-500" : 
              text.length > 800 ? "text-yellow-500" : 
              "text-muted-foreground"
            )}>
              {text.length}/1000 characters
            </span>
          </div>
          
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type what you want your voice to say..."
            className="w-full h-24 p-3 border rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            maxLength={200}
          />
        </div>

        {/* Voice Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
          {/* Speed Control */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Volume2 className="h-4 w-4" />
              <label className="text-sm font-medium">Speed: {speed.toFixed(1)}x</label>
            </div>
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.1"
              value={speed}
              onChange={(e) => setSpeed(parseFloat(e.target.value))}
              className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Slow (0.5x)</span>
              <span>Normal (1.0x)</span>
              <span>Fast (2.0x)</span>
            </div>
          </div>

          {/* Temperature Control */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Thermometer className="h-4 w-4" />
              <label className="text-sm font-medium">Expressiveness: {temperature.toFixed(1)}</label>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Monotone (0.1)</span>
              <span>Natural (0.7)</span>
              <span>Expressive (1.0)</span>
            </div>
          </div>
        </div>

        {/* Generate Button */}
        <div className="mt-6 text-center">
          <button
            onClick={generateSpeech}
            disabled={isGenerating || !text.trim() || text.length > 200 || !appState.modelTrained}
            className={cn(
              "flex items-center gap-2 mx-auto px-6 py-3 rounded-lg font-medium",
              isGenerating 
                ? "bg-blue-400 text-white cursor-not-allowed"
                : !appState.modelTrained || !text.trim() || text.length > 200
                  ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                  : "bg-blue-500 hover:bg-blue-600 text-white"
            )}
          >
            {isGenerating ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Generating Voice...
              </>
            ) : (
              <>
                <Volume2 className="h-5 w-5" />
                Generate Speech
              </>
            )}
          </button>
        </div>

        {/* Status Messages */}
        {error && (
          <div className="flex items-center gap-2 text-red-600 bg-red-50 dark:bg-red-950/20 p-3 rounded-lg mt-4">
            <AlertCircle className="h-4 w-4" />
            {error}
          </div>
        )}

        {success && (
          <div className="flex items-center gap-2 text-green-600 bg-green-50 dark:bg-green-950/20 p-3 rounded-lg mt-4">
            <CheckCircle className="h-4 w-4" />
            {success}
          </div>
        )}
      </div>

      {/* Audio Player */}
      {audioUrl && (
        <div className="bg-card rounded-lg border p-6">
          <h3 className="text-lg font-medium mb-4">Generated Audio</h3>
          
          <div className="flex items-center justify-center gap-4">
            <button
              onClick={playAudio}
              className="flex items-center gap-2 bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg"
            >
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              {isPlaying ? 'Pause' : 'Play'}
            </button>

            <button
              onClick={downloadAudio}
              className="flex items-center gap-2 bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg"
            >
              <Download className="h-4 w-4" />
              Download
            </button>
          </div>

          {/* Audio Element */}
          <audio
            ref={audioRef}
            onEnded={handleAudioEnded}
            onError={handleAudioError}
            style={{ display: 'none' }}
          />

          {/* Audio Waveform Placeholder */}
          <div className="mt-4 h-16 bg-muted/30 rounded-lg flex items-center justify-center">
            <div className="flex gap-1">
              {Array.from({ length: 20 }).map((_, i) => (
                <div
                  key={i}
                  className={cn(
                    "w-1 bg-blue-500 rounded-full transition-all duration-150",
                    isPlaying ? "animate-pulse" : ""
                  )}
                  style={{ height: `${Math.random() * 100 + 20}%` }}
                />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Completion Message */}
      {audioUrl && (
        <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-950/20 dark:to-blue-950/20 rounded-lg border p-6 text-center">
          <h3 className="text-lg font-semibold text-green-600 mb-2">🎉 Congratulations!</h3>
          <p className="text-muted-foreground">
            You've successfully created and tested your personalized AI voice clone! 
            Your voice model is ready for use in applications, demos, and presentations.
          </p>
        </div>
      )}
    </div>
  );
}
