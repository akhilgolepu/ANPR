import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Upload,
  Loader2,
  AlertCircle,
  CheckCircle2,
  Image as ImageIcon,
  Video as VideoIcon,
  FileUp,
} from 'lucide-react';
import { uploadImages, uploadVideo, getJobResults, ANPRResult } from '@/lib/anpr-api';
import ANPRResults from './ANPRResults';

interface PipelineDemoProps {
  onProcessingStart?: () => void;
  onProcessingEnd?: (results: ANPRResult) => void;
}

type FileUploadMode = 'idle' | 'dragging' | 'uploading' | 'completed' | 'error';

export default function PipelineDemo({
  onProcessingStart,
  onProcessingEnd,
}: PipelineDemoProps) {
  const [mode, setMode] = useState<FileUploadMode>('idle');
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [results, setResults] = useState<ANPRResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [processingStage, setProcessingStage] = useState<string | null>(null);

  const waitForCompletion = useCallback(async (initial: ANPRResult) => {
    let latest = initial;
    setProcessingStage(initial.stage ?? 'Queued');
    setUploadProgress(Math.max(initial.progress ?? 5, 5));

    while (latest.status === 'processing') {
      await new Promise(resolve => setTimeout(resolve, 1200));
      latest = await getJobResults(initial.job_id);
      setProcessingStage(latest.stage ?? 'Processing');
      setUploadProgress(Math.max(latest.progress ?? 5, 5));
    }

    return latest;
  }, []);

  // Handle file selection
  const handleFileSelect = useCallback(async (files: File[]) => {
    setError(null);
    setMode('uploading');
    setSelectedFiles(files);
    setUploadProgress(0);
    onProcessingStart?.();
    setProcessingStage(null);

    let progressInterval: ReturnType<typeof setInterval> | null = null;

    try {
      // Validate files
      files.forEach(file => {
        const isImage = file.type.startsWith('image/');
        const isVideo = file.type.startsWith('video/');
        if (!isImage && !isVideo) {
          throw new Error(
            `Invalid file type: ${file.name}. Please upload images or videos.`
          );
        }
      });

      // Process files
      let result: ANPRResult;
      const videoFiles = files.filter(f => f.type.startsWith('video/'));
      const imageFiles = files.filter(f => f.type.startsWith('image/'));

      // Simulate progress
      progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + Math.random() * 20, 95));
      }, 200);

      if (videoFiles.length > 0 && videoFiles.length === files.length) {
        // Process as video
        result = await uploadVideo(videoFiles[0]);
      } else if (imageFiles.length > 0) {
        // Process as images
        result = await uploadImages(imageFiles);
      } else {
        throw new Error('No valid image or video files selected');
      }

      if (progressInterval) {
        clearInterval(progressInterval);
      }

      if (result.status === 'processing') {
        result = await waitForCompletion(result);
      }

      if (result.status === 'error') {
        throw new Error(result.error || 'Processing failed on the server');
      }

      setUploadProgress(100);
      setProcessingStage(result.stage ?? 'Completed');
      await new Promise(resolve => setTimeout(resolve, 500));

      setResults(result);
      setMode('completed');
      onProcessingEnd?.(result);
    } catch (err) {
      if (progressInterval) {
        clearInterval(progressInterval);
      }
      setError(err instanceof Error ? err.message : 'An error occurred');
      setMode('error');
    }
  }, [onProcessingStart, onProcessingEnd, waitForCompletion]);

  // Handle drag and drop
  const handleDragOver = useCallback((e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setMode('dragging');
  }, []);

  const handleDragLeave = useCallback(() => {
    setMode('idle');
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLLabelElement>) => {
      e.preventDefault();
      const files = Array.from(e.dataTransfer.files);
      if (files.length > 0) {
        handleFileSelect(files);
      }
      setMode('idle');
    },
    [handleFileSelect]
  );

  // Handle file input change
  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.currentTarget.files || []);
      if (files.length > 0) {
        handleFileSelect(files);
      }
    },
    [handleFileSelect]
  );

  // Reset handler
  const handleNewUpload = useCallback(() => {
    setMode('idle');
    setSelectedFiles([]);
    setResults(null);
    setError(null);
    setUploadProgress(0);
    setProcessingStage(null);
  }, []);

  // Show results
  if (mode === 'completed' && results) {
    return (
      <section id="results" className="py-24 bg-secondary/30">
        <div className="container mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <ANPRResults 
              results={results} 
              onNewUpload={handleNewUpload}
            />
          </motion.div>
        </div>
      </section>
    );
  }

  return (
    <section id="pipeline" className="py-24 bg-secondary/30">
      <div className="container mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="max-w-2xl mx-auto"
        >
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold mb-4">Try the ANPR System</h2>
            <p className="text-lg text-muted-foreground">
              Upload images or a video to detect and recognize license plates in
              real-time
            </p>
          </div>

          <Card className="glass border-glow">
            <CardHeader>
              <CardTitle>Upload Content</CardTitle>
              <CardDescription>
                Supported formats: JPEG, PNG, MP4, WebM (Max 100 MB)
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-6">
              {/* File Input (hidden) */}
              <input
                type="file"
                multiple
                accept="image/*,video/*"
                onChange={handleInputChange}
                disabled={mode === 'uploading'}
                className="hidden"
                id="file-input"
              />

              {/* Drag and Drop Area */}
              <label
                htmlFor="file-input"
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`relative p-12 rounded-lg border-2 border-dashed transition-all duration-300 cursor-pointer block ${
                  mode === 'dragging'
                    ? 'border-primary bg-primary/5'
                    : 'border-muted-foreground/20 hover:border-primary/50'
                } ${mode === 'uploading' ? 'opacity-50' : ''}`}
              >
                <div className="flex flex-col items-center justify-center">
                  <div className="mb-4">
                    <FileUp className="w-12 h-12 text-primary mx-auto" />
                  </div>
                  <p className="font-semibold text-foreground mb-1">
                    {mode === 'dragging'
                      ? 'Drop files here'
                      : 'Drag and drop files here'}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    or click to select images/videos
                  </p>
                </div>
              </label>

              {/* File Selection Button */}
              {mode === 'idle' && (
                <Button
                  className="w-full glow-orange"
                  onClick={() => {
                    const input = document.getElementById(
                      'file-input'
                    ) as HTMLInputElement;
                    input?.click();
                  }}
                >
                  <Upload className="w-4 h-4 mr-2" />
                  Select Files
                </Button>
              )}

              {/* Selected Files Display */}
              {selectedFiles.length > 0 && (
                <div className="space-y-3">
                  <p className="text-sm font-medium text-foreground">
                    Selected {selectedFiles.length} file{selectedFiles.length !== 1 ? 's' : ''}:
                  </p>
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {selectedFiles.map((file, idx) => (
                      <div
                        key={idx}
                        className="flex items-center gap-3 p-3 bg-secondary rounded-lg"
                      >
                        {file.type.startsWith('video/') ? (
                          <VideoIcon className="w-4 h-4 text-blue-500 flex-shrink-0" />
                        ) : (
                          <ImageIcon className="w-4 h-4 text-green-500 flex-shrink-0" />
                        )}
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium truncate">
                            {file.name}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {(file.size / 1024 / 1024).toFixed(2)} MB
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Upload Progress */}
              {mode === 'uploading' && (
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <Loader2 className="w-5 h-5 animate-spin text-primary" />
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium text-foreground">
                        Processing {selectedFiles.length} file{selectedFiles.length !== 1 ? 's' : ''}...
                      </p>
                      {processingStage && (
                        <p className="text-xs text-muted-foreground mt-1">{processingStage}</p>
                      )}
                    </div>
                  </div>
                  <Progress value={uploadProgress} className="h-2" />
                  <p className="text-xs text-muted-foreground text-right">
                    {Math.round(uploadProgress)}%
                  </p>
                </div>
              )}

              {/* Error Display */}
              {mode === 'error' && error && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                  <Button
                    variant="outline"
                    size="sm"
                    className="mt-3"
                    onClick={handleNewUpload}
                  >
                    Try Again
                  </Button>
                </Alert>
              )}

              {/* Success Message */}
              {mode === 'completed' && !error && (
                <Alert className="border-success/50 bg-success/5">
                  <CheckCircle2 className="h-4 w-4 text-success" />
                  <AlertDescription className="text-success">
                    Processing completed successfully!
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Info Cards */}
          <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <Card className="glass">
                <CardHeader>
                  <ImageIcon className="w-5 h-5 text-primary mb-2" />
                  <CardTitle className="text-base">Image Support</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Process single or multiple images (JPEG, PNG, WebP)
                  </p>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Card className="glass">
                <CardHeader>
                  <VideoIcon className="w-5 h-5 text-primary mb-2" />
                  <CardTitle className="text-base">Video Support</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Process video files (MP4, WebM) with frame extraction
                  </p>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <Card className="glass">
                <CardHeader>
                  <CheckCircle2 className="w-5 h-5 text-success mb-2" />
                  <CardTitle className="text-base">Real-time Results</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    See detection & recognition results instantly
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
