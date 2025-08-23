import React, { useState, useEffect } from 'react';
import { Activity, Server, HardDrive, MemoryStick } from 'lucide-react';
import { cn } from '@/lib/utils';

interface HealthData {
  python_version: string;
  torch_version: string;
  memory_available_gb: number;
  disk_free_gb: number;
  model_loaded: boolean;
  status: string;
  platform: string;
}

export function SystemStatus() {
  const [healthData, setHealthData] = useState<HealthData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHealthData = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch('http://127.0.0.1:8000/healthz');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setHealthData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch health data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealthData();
    const interval = setInterval(fetchHealthData, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const getMemoryStatus = (memoryGb: number) => {
    if (memoryGb > 4) return { color: 'text-green-500', label: 'Good' };
    if (memoryGb > 2) return { color: 'text-yellow-500', label: 'Low' };
    return { color: 'text-red-500', label: 'Critical' };
  };

  if (loading) {
    return (
      <div className="bg-card rounded-lg border p-6">
        <div className="flex items-center gap-2 mb-4">
          <Activity className="h-5 w-5 animate-pulse" />
          <h3 className="text-lg font-semibold">System Status</h3>
        </div>
        <div className="text-muted-foreground">Loading system information...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-card rounded-lg border p-6">
        <div className="flex items-center gap-2 mb-4">
          <Activity className="h-5 w-5 text-red-500" />
          <h3 className="text-lg font-semibold">System Status</h3>
        </div>
        <div className="text-red-500 mb-4">Error: {error}</div>
        <button
          onClick={fetchHealthData}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!healthData) return null;

  const memoryStatus = getMemoryStatus(healthData.memory_available_gb);

  return (
    <div className="bg-card rounded-lg border p-6">
      <div className="flex items-center gap-2 mb-4">
        <Activity className={cn("h-5 w-5", healthData.status === 'healthy' ? 'text-green-500' : 'text-red-500')} />
        <h3 className="text-lg font-semibold">System Status</h3>
        <span className={cn(
          "px-2 py-1 rounded-full text-xs font-medium",
          healthData.status === 'healthy' 
            ? 'bg-green-100 text-green-800' 
            : 'bg-red-100 text-red-800'
        )}>
          {healthData.status}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <Server className="h-4 w-4 text-muted-foreground" />
            <div>
              <div className="text-sm font-medium">Backend</div>
              <div className="text-xs text-muted-foreground">Python {healthData.python_version}</div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <MemoryStick className={cn("h-4 w-4", memoryStatus.color)} />
            <div>
              <div className="text-sm font-medium flex items-center gap-2">
                Memory Available
                <span className={cn("text-xs", memoryStatus.color)}>({memoryStatus.label})</span>
              </div>
              <div className="text-xs text-muted-foreground">{healthData.memory_available_gb} GB</div>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <HardDrive className="h-4 w-4 text-muted-foreground" />
            <div>
              <div className="text-sm font-medium">Disk Space</div>
              <div className="text-xs text-muted-foreground">{healthData.disk_free_gb} GB free</div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Activity className="h-4 w-4 text-muted-foreground" />
            <div>
              <div className="text-sm font-medium">PyTorch</div>
              <div className="text-xs text-muted-foreground">v{healthData.torch_version}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t">
        <div className="text-xs text-muted-foreground">
          Platform: {healthData.platform} | Model Loaded: {healthData.model_loaded ? 'Yes' : 'No'}
        </div>
      </div>
    </div>
  );
}
