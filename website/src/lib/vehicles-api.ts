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

export const listVehicles = (params?: { status?: string; search?: string }) => {
  const qs = new URLSearchParams();
  if (params?.status) qs.set("status", params.status);
  if (params?.search) qs.set("search", params.search);
  const query = qs.toString() ? `?${qs}` : "";
  return api<VehicleRecord[]>(`/api/vehicles${query}`);
};

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
