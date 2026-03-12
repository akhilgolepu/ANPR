/**
 * ANPR API Types and Client
 * Handles communication with the FastAPI backend
 */

export interface PlateDetection {
  plate_text: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  vehicle_crop_url: string;
  plate_crop_url: string;
}

export interface ANPRResult {
  job_id: string;
  status: 'processing' | 'completed' | 'error';
  input_type: 'image' | 'video';
  total_detections: number;
  processing_time: number;
  detections: PlateDetection[];
  output_file_url?: string;
  error?: string;
}

const API_BASE = 'http://localhost:8000/api';
export const API_HOST = 'http://localhost:8000';

/**
 * Resolve a relative static asset path from the backend to a full URL
 */
export function getAssetUrl(path: string): string {
  if (!path) return '';
  if (path.startsWith('http')) return path;
  return `${API_HOST}${path}`;
}

/**
 * Upload a single or multiple images for ANPR processing
 */
export async function uploadImages(files: File[]): Promise<ANPRResult> {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));

  const response = await fetch(`${API_BASE}/process-images`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Failed to upload images: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Upload a video file for ANPR processing
 */
export async function uploadVideo(file: File): Promise<ANPRResult> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/process-video`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Failed to upload video: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Poll for job results
 */
export async function getJobResults(jobId: string): Promise<ANPRResult> {
  const response = await fetch(`${API_BASE}/results/${jobId}`);

  if (!response.ok) {
    throw new Error(`Failed to fetch results: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Export results as JSON
 */
export function exportResultsAsJSON(results: ANPRResult, filename?: string) {
  const json = JSON.stringify(results, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename || `anpr-results-${results.job_id}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Export results as CSV
 */
export function exportResultsAsCSV(results: ANPRResult, filename?: string) {
  let csv = 'Detection #,License Plate,Confidence,Bbox (x1, y1, x2, y2)\n';
  results.detections.forEach((det, idx) => {
    csv += `${idx + 1},"${det.plate_text}",${(det.confidence * 100).toFixed(1)}%,"${det.bbox.join(', ')}"\n`;
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename || `anpr-results-${results.job_id}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
