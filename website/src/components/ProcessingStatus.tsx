import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Loader2, 
  CheckCircle, 
  AlertCircle, 
  X,
  Clock,
  Cpu,
  Eye
} from 'lucide-react';
import { ProcessingStatus as ProcessingStatusType } from '@/lib/anpr-api';

interface ProcessingStatusProps {
  status: ProcessingStatusType;
  onCancel?: () => void;
  elapsedTime?: number;
}

export default function ProcessingStatus({ status, onCancel, elapsedTime }: ProcessingStatusProps) {
  const getStatusIcon = () => {
    switch (status.status) {
      case 'processing':
        return <Loader2 className="w-6 h-6 text-primary animate-spin" />;
      case 'completed':
        return <CheckCircle className="w-6 h-6 text-success" />;
      case 'error':
        return <AlertCircle className="w-6 h-6 text-destructive" />;
      default:
        return <Clock className="w-6 h-6 text-muted-foreground" />;
    }
  };

  const getStatusColor = () => {
    switch (status.status) {
      case 'processing':
        return 'default';
      case 'completed':
        return 'success';
      case 'error':
        return 'destructive';
      default:
        return 'secondary';
    }
  };

  const getProcessingStage = () => {
    if (status.progress <= 0.2) return 'Loading and validating files...';
    if (status.progress <= 0.4) return 'Detecting vehicles...';
    if (status.progress <= 0.6) return 'Extracting license plates...';
    if (status.progress <= 0.8) return 'Running OCR recognition...';
    if (status.progress <= 0.9) return 'Saving results...';
    return 'Finalizing...';
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className="w-full max-w-2xl mx-auto"
    >
      <Card className="glass border-glow">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {getStatusIcon()}
              <div>
                <CardTitle className="text-lg">Processing Status</CardTitle>
                <p className="text-sm text-muted-foreground">
                  Job ID: {status.job_id.slice(0, 8)}...
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <Badge variant={getStatusColor() as any} className="capitalize">
                {status.status}
              </Badge>
              
              {onCancel && status.status === 'processing' && (
                <Button 
                  variant="ghost" 
                  size="sm"
                  onClick={onCancel}
                  className="text-destructive hover:text-destructive/80"
                >
                  <X className="w-4 h-4 mr-1" />
                  Cancel
                </Button>
              )}
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Progress Bar */}
          {status.status === 'processing' && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Progress</span>
                <span className="font-mono">{Math.round(status.progress * 100)}%</span>
              </div>
              
              <Progress 
                value={status.progress * 100} 
                className="h-2"
              />
              
              {/* Processing Stage Indicator */}
              <motion.div 
                key={getProcessingStage()}
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center gap-2 text-sm text-primary"
              >
                <Cpu className="w-4 h-4" />
                {getProcessingStage()}
              </motion.div>
            </div>
          )}

          {/* Status Message */}
          <div className="rounded-lg bg-muted/50 p-4">
            <div className="flex items-start gap-3">
              <Eye className="w-5 h-5 text-muted-foreground mt-0.5" />
              <div>
                <p className="font-medium text-sm">Status Update</p>
                <p className="text-muted-foreground text-sm mt-1">
                  {status.message}
                </p>
              </div>
            </div>
          </div>

          {/* Processing Time */}
          {elapsedTime !== undefined && (
            <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
              <Clock className="w-4 h-4" />
              Processing time: {elapsedTime.toFixed(1)}s
            </div>
          )}

          {/* Processing Animation */}
          {status.status === 'processing' && (
            <div className="space-y-3">
              <div className="text-center text-sm text-muted-foreground">
                Analyzing your content with AI models...
              </div>
              
              {/* Animated processing stages */}
              <div className="grid grid-cols-3 gap-2">
                {[
                  { label: 'Vehicle Detection', progress: Math.max(0, Math.min(1, (status.progress - 0.1) * 3)) },
                  { label: 'Plate Extraction', progress: Math.max(0, Math.min(1, (status.progress - 0.4) * 3)) },
                  { label: 'OCR Recognition', progress: Math.max(0, Math.min(1, (status.progress - 0.7) * 3)) },
                ].map((stage, index) => (
                  <div key={stage.label} className="text-center">
                    <div className="text-xs text-muted-foreground mb-1">
                      {stage.label}
                    </div>
                    <div className="h-1 bg-muted rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${stage.progress * 100}%` }}
                        transition={{ duration: 0.5 }}
                        className="h-full bg-gradient-to-r from-primary/50 to-primary"
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}