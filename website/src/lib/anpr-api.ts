/**
 * ANPR API Types and Client
 * Handles communication with the FastAPI backend
 */

export interface VehicleMatchSummary {
  plate_number: string;
  status: 'Clear' | 'Stolen/Missing' | 'Recovered';
  vehicle_make: string;
  vehicle_model: string;
  owner_name: string;
  registered_rto_state: string;
  registered_rto_code: string;
  police_complaint_id?: string | null;
}

export interface PlateDetection {
  plate_text: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  vehicle_crop_url: string;
  plate_crop_url: string;
  raw_ocr_text?: string;  // Raw TrOCR output before cleaning
  ocr_engine?: string;    // e.g. "trocr"
  top_ocr_candidates?: string[];
  format_score?: number;
  review_required?: boolean;
  registry_match?: VehicleMatchSummary | null;
  human_corrected_text?: string | null;
  human_verified?: boolean;
  seen_count?: number;
  first_seen_sec?: number | null;
  last_seen_sec?: number | null;
  source_frame?: number | null;
  source_file_name?: string | null;
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
  progress?: number;
  stage?: string | null;
  alert_count?: number;
  review_count?: number;
}

const _apiHost = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000';
const API_BASE = `${_apiHost}/api`;
export const API_HOST = _apiHost;

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

export async function correctDetection(jobId: string, detectionIndex: number, correctedText: string): Promise<ANPRResult> {
  const response = await fetch(`${API_BASE}/results/${jobId}/detections/${detectionIndex}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ corrected_text: correctedText }),
  });

  if (!response.ok) {
    throw new Error(`Failed to save correction: ${response.statusText}`);
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
  let csv = 'Detection #,License Plate,Corrected Plate,Confidence,Format Score,Registry Status,Bbox (x1, y1, x2, y2)\n';
  results.detections.forEach((det, idx) => {
    csv += `${idx + 1},"${det.plate_text}","${det.human_corrected_text ?? ''}",${(det.confidence * 100).toFixed(1)}%,${((det.format_score ?? 0) * 100).toFixed(1)}%,"${det.registry_match?.status ?? 'Unmatched'}","${det.bbox.join(', ')}"\n`;
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
