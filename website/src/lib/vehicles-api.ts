import { API_HOST } from "./anpr-api";

// ── Types ─────────────────────────────────────────────────────────────────────

export type VehicleStatus = "Clear" | "Stolen/Missing" | "Recovered";
export type VehicleType = "Car" | "Bike" | "Truck" | "Bus" | "Auto" | "Other";

export interface VehicleRecord {
  plate_number: string;
  vehicle_make: string;
  vehicle_model: string;
  vehicle_year: number | null;
  vehicle_color: string;
  vehicle_type: VehicleType;
  owner_name: string;
  owner_phone: string | null;
  owner_email: string | null;
  owner_address: string | null;
  registered_rto_state: string;
  registered_rto_code: string;
  chassis_number: string | null;
  engine_number: string | null;
  registration_date: string | null;   // ISO date string
  registration_expiry: string | null;
  insurance_expiry: string | null;
  status: VehicleStatus;
  police_complaint_id: string | null;
  missing_date: string | null;        // ISO datetime string
  recovery_date: string | null;
  created_at: string;
  updated_at: string;
}

export interface VehicleCreate {
  plate_number: string;
  vehicle_make: string;
  vehicle_model: string;
  vehicle_year?: number | null;
  vehicle_color: string;
  vehicle_type?: VehicleType;
  owner_name: string;
  owner_phone?: string | null;
  owner_email?: string | null;
  owner_address?: string | null;
  registered_rto_state: string;
  registered_rto_code: string;
  chassis_number?: string | null;
  engine_number?: string | null;
  registration_date?: string | null;
  registration_expiry?: string | null;
  insurance_expiry?: string | null;
}

export type VehicleUpdate = Partial<Omit<VehicleCreate, "plate_number">>;

export interface ComplaintRequest {
  complaint_id: string;
  reported_by?: string;
  reporting_station?: string;
  theft_location?: string;
}

export interface RecoveryRequest {
  resolution_notes?: string;
  officer?: string;
}

export interface ActionResponse {
  success: boolean;
  message: string;
  vehicle: VehicleRecord | null;
}

export interface BulkImportResponse {
  success: boolean;
  imported: number;
  updated: number;
  errors: string[];
}

// ── Fetch helpers ─────────────────────────────────────────────────────────────

async function api<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(`${API_HOST}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => String(res.status));
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── API functions ─────────────────────────────────────────────────────────────

export const listVehicles = (params?: { status?: string; search?: string; vehicleType?: string; stateCode?: string }) => {
  const qs = new URLSearchParams();
  if (params?.status) qs.set("status", params.status);
  if (params?.search) qs.set("search", params.search);
  if (params?.vehicleType) qs.set("vehicle_type", params.vehicleType);
  if (params?.stateCode) qs.set("state_code", params.stateCode);
  const query = qs.toString() ? `?${qs}` : "";
  return api<VehicleRecord[]>(`/api/vehicles${query}`);
};

export async function exportVehiclesCsv(params?: { status?: string; search?: string; vehicleType?: string; stateCode?: string }) {
  const qs = new URLSearchParams();
  if (params?.status) qs.set("status", params.status);
  if (params?.search) qs.set("search", params.search);
  if (params?.vehicleType) qs.set("vehicle_type", params.vehicleType);
  if (params?.stateCode) qs.set("state_code", params.stateCode);
  const query = qs.toString() ? `?${qs}` : "";
  const res = await fetch(`${API_HOST}/api/vehicles/export${query}`);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "vehicle-registry.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function importVehiclesCsv(file: File): Promise<BulkImportResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_HOST}/api/vehicles/import`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => String(res.status));
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<BulkImportResponse>;
}

export const getVehicle = (plate: string) =>
  api<VehicleRecord>(`/api/vehicles/${encodeURIComponent(plate)}`);

export const createVehicle = (payload: VehicleCreate) =>
  api<ActionResponse>("/api/vehicles", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const updateVehicle = (plate: string, payload: VehicleUpdate) =>
  api<ActionResponse>(`/api/vehicles/${encodeURIComponent(plate)}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });

export const deleteVehicle = (plate: string) =>
  api<ActionResponse>(`/api/vehicles/${encodeURIComponent(plate)}`, {
    method: "DELETE",
  });

export const fileComplaint = (plate: string, payload: ComplaintRequest) =>
  api<ActionResponse>(`/api/vehicles/${encodeURIComponent(plate)}/file-complaint`, {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const markRecovered = (plate: string, payload: RecoveryRequest) =>
  api<ActionResponse>(`/api/vehicles/${encodeURIComponent(plate)}/mark-recovered`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
