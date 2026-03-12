import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { 
  Download, 
  Car, 
  Hash, 
  Clock, 
  Eye,
  FileText,
  CheckCircle,
  AlertCircle,
  ShieldAlert,
  SearchCheck,
  PencilLine,
  GalleryHorizontalEnd,
} from 'lucide-react';
import { ANPRResult, PlateDetection, correctDetection, exportResultsAsJSON, exportResultsAsCSV, getAssetUrl } from '@/lib/anpr-api';

interface ANPRResultsProps {
  results: ANPRResult;
  onDownload?: () => void;
  onNewUpload?: () => void;
}

export default function ANPRResults({ results, onDownload, onNewUpload }: ANPRResultsProps) {
  const [currentResults, setCurrentResults] = useState(results);

  useEffect(() => {
    setCurrentResults(results);
  }, [results]);

  const handleDownloadJSON = () => {
    exportResultsAsJSON(currentResults);
  };

  const handleDownloadCSV = () => {
    exportResultsAsCSV(currentResults);
  };

  const handleCorrection = async (index: number, correctedText: string) => {
    const updated = await correctDetection(currentResults.job_id, index, correctedText);
    setCurrentResults(updated);
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
                  Found {currentResults.total_detections} license plate{currentResults.total_detections !== 1 ? 's' : ''} 
                  {currentResults.processing_time > 0 && ` in ${currentResults.processing_time.toFixed(2)}s`}
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
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4">
        <Card className="glass">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <Car className="w-5 h-5 text-primary" />
              <div>
                <p className="text-sm text-muted-foreground">Input Type</p>
                <p className="font-semibold capitalize">{currentResults.input_type}</p>
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
                <p className="font-semibold">{currentResults.total_detections}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="glass">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <ShieldAlert className="w-5 h-5 text-red-400" />
              <div>
                <p className="text-sm text-muted-foreground">Alerts</p>
                <p className="font-semibold">{currentResults.alert_count ?? 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="glass">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <SearchCheck className="w-5 h-5 text-warning" />
              <div>
                <p className="text-sm text-muted-foreground">Needs Review</p>
                <p className="font-semibold">{currentResults.review_count ?? 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {currentResults.processing_time > 0 && (
          <Card className="glass">
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <Clock className="w-5 h-5 text-primary" />
                <div>
                  <p className="text-sm text-muted-foreground">Processing Time</p>
                  <p className="font-semibold">{currentResults.processing_time.toFixed(2)}s</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Individual Detections */}
      {currentResults.detections.length > 0 ? (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Eye className="w-5 h-5" />
            Detected License Plates
          </h3>
          
          <AnimatePresence>
            {currentResults.detections.map((detection, index) => (
              <DetectionCard 
                key={`${currentResults.job_id}-${index}`}
                jobId={currentResults.job_id}
                detection={detection}
                index={index}
                onCorrect={handleCorrection}
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
      {currentResults.detections.length > 0 && (
        <Card className="glass">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <GalleryHorizontalEnd className="w-5 h-5" />
              Detection Gallery
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {currentResults.detections.map((detection, index) => {
                const effectivePlate = detection.human_corrected_text || detection.plate_text;
                return (
                  <div key={`gallery-${currentResults.job_id}-${index}`} className="rounded-xl border border-border bg-muted/20 p-4 space-y-3">
                    <div className="flex items-center justify-between gap-3">
                      <div className="font-mono font-semibold text-sm tracking-wider">{effectivePlate || '—'}</div>
                      {detection.registry_match && (
                        <Badge variant={detection.registry_match.status === 'Stolen/Missing' ? 'destructive' : 'outline'}>
                          {detection.registry_match.status}
                        </Badge>
                      )}
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <img src={getAssetUrl(detection.vehicle_crop_url)} alt={`Vehicle ${index + 1}`} className="w-full aspect-video rounded-lg object-cover" loading="lazy" />
                      <img src={getAssetUrl(detection.plate_crop_url)} alt={`Plate ${effectivePlate}`} className="w-full aspect-[3/1] rounded-lg object-cover bg-muted" loading="lazy" />
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {currentResults.output_file_url && (
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
                src={getAssetUrl(currentResults.output_file_url!)}
              >
                Your browser does not support video playback.
              </video>
            </div>
            <div className="mt-4 flex justify-center">
              <Button asChild variant="outline">
                <a 
                  href={getAssetUrl(currentResults.output_file_url!)}
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
  jobId: string;
  detection: PlateDetection;
  index: number;
  onCorrect: (index: number, correctedText: string) => Promise<void>;
}

function DetectionCard({ jobId, detection, index, onCorrect }: DetectionCardProps) {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [draftCorrection, setDraftCorrection] = useState(detection.human_corrected_text || detection.plate_text);
  const [saving, setSaving] = useState(false);
  const confidenceColor = detection.confidence >= 0.8 ? 'success' : 
                         detection.confidence >= 0.6 ? 'warning' : 'destructive';
  const effectivePlate = detection.human_corrected_text || detection.plate_text || '—';
  const matchStatus = detection.registry_match?.status;

  const saveCorrection = async () => {
    setSaving(true);
    try {
      await onCorrect(index, draftCorrection);
      setDialogOpen(false);
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
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
                  {jobId.slice(0, 8)} · License Plate
                </Badge>
                <div className="text-3xl font-mono font-bold text-gradient-accent">
                  {effectivePlate}
                </div>
                {detection.human_corrected_text && detection.human_corrected_text !== detection.plate_text && (
                  <div className="text-xs text-muted-foreground mt-2 font-mono">
                    Model read: {detection.plate_text}
                  </div>
                )}
              </div>
              
              <Separator />
              
              <div className="space-y-2">
                <div className="flex justify-center gap-2 flex-wrap">
                  <Badge variant={confidenceColor as any}>
                    {(detection.confidence * 100).toFixed(1)}% confidence
                  </Badge>
                  <Badge variant="outline">
                    {(100 * (detection.format_score ?? 0)).toFixed(0)}% format score
                  </Badge>
                  {detection.ocr_engine && (
                    <Badge variant="outline" className="text-xs">
                      {detection.ocr_engine.toUpperCase()}
                    </Badge>
                  )}
                  {detection.review_required && (
                    <Badge variant="outline" className="text-warning border-warning/40">Review</Badge>
                  )}
                  {detection.human_verified && (
                    <Badge variant="outline" className="text-green-400 border-green-500/40">Human Verified</Badge>
                  )}
                  {matchStatus && (
                    <Badge variant={matchStatus === 'Stolen/Missing' ? 'destructive' : 'outline'}>
                      {matchStatus}
                    </Badge>
                  )}
                </div>

                {detection.raw_ocr_text && detection.raw_ocr_text !== detection.plate_text && (
                  <div className="text-xs text-muted-foreground font-mono bg-muted/50 rounded px-2 py-1">
                    Raw: "{detection.raw_ocr_text}"
                  </div>
                )}

                {detection.top_ocr_candidates && detection.top_ocr_candidates.length > 0 && (
                  <div className="flex justify-center gap-2 flex-wrap">
                    {detection.top_ocr_candidates.map((candidate) => (
                      <Badge key={candidate} variant="secondary" className="font-mono text-xs">
                        {candidate}
                      </Badge>
                    ))}
                  </div>
                )}

                {detection.registry_match && (
                  <div className="text-sm text-muted-foreground bg-muted/40 rounded-lg px-3 py-2">
                    <span className="font-semibold text-foreground">{detection.registry_match.vehicle_make} {detection.registry_match.vehicle_model}</span>
                    {' · '}
                    {detection.registry_match.owner_name}
                    {' · '}
                    {detection.registry_match.registered_rto_state}
                    {detection.registry_match.police_complaint_id && ` · FIR ${detection.registry_match.police_complaint_id}`}
                  </div>
                )}

                <div className="text-sm text-muted-foreground">
                  Position: [{detection.bbox[0]}, {detection.bbox[1]}] →{' '}
                  [{detection.bbox[2]}, {detection.bbox[3]}]
                </div>

                {(detection.seen_count ?? 1) > 1 && (
                  <div className="text-sm text-muted-foreground">
                    Seen {(detection.seen_count ?? 1)} times · {detection.first_seen_sec?.toFixed(1)}s → {detection.last_seen_sec?.toFixed(1)}s
                  </div>
                )}

                <div className="flex justify-center">
                  <Button variant="outline" size="sm" onClick={() => setDialogOpen(true)}>
                    <PencilLine className="w-4 h-4 mr-2" />
                    Correct Plate
                  </Button>
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

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>Correct Detection</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <div className="text-xs text-muted-foreground font-mono">Current: {effectivePlate}</div>
            <Input value={draftCorrection} onChange={(e) => setDraftCorrection(e.target.value.toUpperCase())} />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDialogOpen(false)} disabled={saving}>Cancel</Button>
            <Button onClick={saveCorrection} disabled={saving || !draftCorrection.trim()}>
              {saving ? 'Saving...' : 'Save Correction'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}