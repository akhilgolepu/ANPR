import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { 
  Download, 
  Car, 
  Hash, 
  Clock, 
  Eye,
  FileText,
  CheckCircle,
  AlertCircle
} from 'lucide-react';
import { ANPRResult, PlateDetection, exportResultsAsJSON, exportResultsAsCSV, getAssetUrl } from '@/lib/anpr-api';

interface ANPRResultsProps {
  results: ANPRResult;
  onDownload?: () => void;
  onNewUpload?: () => void;
}

export default function ANPRResults({ results, onDownload, onNewUpload }: ANPRResultsProps) {
  const handleDownloadJSON = () => {
    exportResultsAsJSON(results);
  };

  const handleDownloadCSV = () => {
    exportResultsAsCSV(results);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="w-full max-w-7xl mx-auto space-y-6"
    >
      {/* Results Header */}
      <Card className="glass border-glow">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-success/20 rounded-lg">
                <CheckCircle className="w-6 h-6 text-success" />
              </div>
              <div>
                <CardTitle className="text-xl">Processing Complete</CardTitle>
                <p className="text-muted-foreground">
                  Found {results.total_detections} license plate{results.total_detections !== 1 ? 's' : ''} 
                  {results.processing_time > 0 && ` in ${results.processing_time.toFixed(2)}s`}
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              {onDownload && (
                <Button onClick={onDownload} variant="outline" size="sm">
                  <Download className="w-4 h-4 mr-2" />
                  Export
                </Button>
              )}
              <Button onClick={handleDownloadJSON} variant="outline" size="sm">
                <Download className="w-4 h-4 mr-2" />
                JSON
              </Button>
              <Button onClick={handleDownloadCSV} variant="outline" size="sm">
                <Download className="w-4 h-4 mr-2" />
                CSV
              </Button>
              {onNewUpload && (
                <Button onClick={onNewUpload} className="glow-orange" size="sm">
                  <FileText className="w-4 h-4 mr-2" />
                  New Upload
                </Button>
              )}
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Results Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="glass">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <Car className="w-5 h-5 text-primary" />
              <div>
                <p className="text-sm text-muted-foreground">Input Type</p>
                <p className="font-semibold capitalize">{results.input_type}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="glass">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <Hash className="w-5 h-5 text-primary" />
              <div>
                <p className="text-sm text-muted-foreground">Detections</p>
                <p className="font-semibold">{results.total_detections}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {results.processing_time > 0 && (
          <Card className="glass">
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <Clock className="w-5 h-5 text-primary" />
                <div>
                  <p className="text-sm text-muted-foreground">Processing Time</p>
                  <p className="font-semibold">{results.processing_time.toFixed(2)}s</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Individual Detections */}
      {results.detections.length > 0 ? (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Eye className="w-5 h-5" />
            Detected License Plates
          </h3>
          
          <AnimatePresence>
            {results.detections.map((detection, index) => (
              <DetectionCard 
                key={`${results.job_id}-${index}`}
                detection={detection}
                index={index}
              />
            ))}
          </AnimatePresence>
        </div>
      ) : (
        <Card className="glass border-warning/50">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3 text-warning">
              <AlertCircle className="w-6 h-6" />
              <div>
                <p className="font-semibold">No License Plates Detected</p>
                <p className="text-sm text-muted-foreground">
                  The analysis completed but no license plates were found in the uploaded content.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Video Output (if available) */}
      {results.output_file_url && (
        <Card className="glass">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5" />
              Processed Video
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="aspect-video bg-muted rounded-lg overflow-hidden">
              <video 
                controls 
                className="w-full h-full object-cover"
                src={getAssetUrl(results.output_file_url!)}
              >
                Your browser does not support video playback.
              </video>
            </div>
            <div className="mt-4 flex justify-center">
              <Button asChild variant="outline">
                <a 
                  href={getAssetUrl(results.output_file_url!)}
                  download
                  className="flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Download Video
                </a>
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </motion.div>
  );
}

interface DetectionCardProps {
  detection: PlateDetection;
  index: number;
}

function DetectionCard({ detection, index }: DetectionCardProps) {
  const confidenceColor = detection.confidence >= 0.8 ? 'success' : 
                         detection.confidence >= 0.6 ? 'warning' : 'destructive';

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.1, duration: 0.4 }}
    >
      <Card className="glass">
        <CardContent className="pt-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-center">
            
            {/* Vehicle Image (Left) */}
            <div className="order-1">
              <div className="aspect-video bg-muted rounded-lg overflow-hidden">
                <img
                  src={getAssetUrl(detection.vehicle_crop_url)}
                  alt={`Vehicle ${index + 1}`}
                  className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
                  loading="lazy"
                />
              </div>
              <p className="text-sm text-muted-foreground mt-2 text-center">
                Vehicle Detection #{index + 1}
              </p>
            </div>

            {/* Plate Text and Details (Center) */}
            <div className="order-2 text-center space-y-4">
              <div>
                <Badge 
                  variant="outline" 
                  className="text-xs text-muted-foreground mb-3"
                >
                  License Plate
                </Badge>
                <div className="text-3xl font-mono font-bold text-gradient-accent">
                  {detection.plate_text}
                </div>
              </div>
              
              <Separator />
              
              <div className="space-y-2">
                <div className="flex justify-center gap-2 flex-wrap">
                  <Badge variant={confidenceColor as any}>
                    {(detection.confidence * 100).toFixed(1)}% confidence
                  </Badge>
                  {detection.ocr_engine && (
                    <Badge variant="outline" className="text-xs">
                      {detection.ocr_engine.toUpperCase()}
                    </Badge>
                  )}
                </div>

                {detection.raw_ocr_text && detection.raw_ocr_text !== detection.plate_text && (
                  <div className="text-xs text-muted-foreground font-mono bg-muted/50 rounded px-2 py-1">
                    Raw: "{detection.raw_ocr_text}"
                  </div>
                )}

                <div className="text-sm text-muted-foreground">
                  Position: [{detection.bbox[0]}, {detection.bbox[1]}] →{' '}
                  [{detection.bbox[2]}, {detection.bbox[3]}]
                </div>
              </div>
            </div>

            {/* Plate Image (Right) */}
            <div className="order-3">
              <div className="aspect-[3/1] bg-muted rounded-lg overflow-hidden">
                <img
                  src={getAssetUrl(detection.plate_crop_url)}
                  alt={`Plate: ${detection.plate_text}`}
                  className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
                  loading="lazy"
                />
              </div>
              <p className="text-sm text-muted-foreground mt-2 text-center">
                License Plate Crop
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}