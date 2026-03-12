import { useEffect, useState, useCallback } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Shield, ArrowLeft, Plus, Search, Edit2, Trash2,
  AlertTriangle, CheckCircle2, Car, RefreshCw,
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Dialog, DialogContent, DialogHeader,
  DialogTitle, DialogFooter,
} from "@/components/ui/dialog";
import {
  Select, SelectContent, SelectItem,
  SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/components/ui/use-toast";
import {
  listVehicles, createVehicle, updateVehicle, deleteVehicle,
  fileComplaint, markRecovered,
  VehicleRecord, VehicleCreate, VehicleUpdate,
  ComplaintRequest, RecoveryRequest, VehicleType,
} from "@/lib/vehicles-api";

// ─── helpers ─────────────────────────────────────────────────────────────────

const STATUS_COLOR: Record<string, string> = {
  "Clear": "bg-green-500/20 text-green-400 border-green-500/30",
  "Stolen/Missing": "bg-red-500/20 text-red-400 border-red-500/30",
  "Recovered": "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
};

const VEHICLE_TYPES: VehicleType[] = ["Car", "Bike", "Truck", "Bus", "Auto", "Other"];

function statusBadge(status: string) {
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs border font-medium ${STATUS_COLOR[status] ?? ""}`}>
      {status === "Clear" && <CheckCircle2 className="h-3 w-3" />}
      {status === "Stolen/Missing" && <AlertTriangle className="h-3 w-3" />}
      {status === "Recovered" && <Car className="h-3 w-3" />}
      {status}
    </span>
  );
}

// ─── empty form state ─────────────────────────────────────────────────────────

function emptyCreate(): VehicleCreate {
  return {
    plate_number: "",
    vehicle_make: "",
    vehicle_model: "",
    vehicle_color: "",
    vehicle_type: "Car",
    owner_name: "",
    registered_rto_state: "",
    registered_rto_code: "",
    vehicle_year: null,
    owner_phone: "",
    owner_email: "",
    owner_address: "",
    chassis_number: "",
    engine_number: "",
    registration_date: "",
    registration_expiry: "",
    insurance_expiry: "",
  };
}

// ─── VehicleForm (shared for Add + Edit) ─────────────────────────────────────

interface VehicleFormProps {
  values: VehicleCreate;
  onChange: (v: VehicleCreate) => void;
  readonlyPlate?: boolean;
}

function VehicleForm({ values, onChange, readonlyPlate }: VehicleFormProps) {
  const set = (key: keyof VehicleCreate, val: string | number | null) =>
    onChange({ ...values, [key]: val });

  const field = (
    label: string,
    key: keyof VehicleCreate,
    type: string = "text",
    placeholder?: string,
  ) => (
    <div className="space-y-1">
      <Label className="text-xs text-muted-foreground">{label}</Label>
      <Input
        type={type}
        placeholder={placeholder}
        value={(values[key] as string) ?? ""}
        readOnly={key === "plate_number" && readonlyPlate}
        className={`h-8 text-sm bg-background ${key === "plate_number" && readonlyPlate ? "opacity-60" : ""}`}
        onChange={(e) =>
          set(key, type === "number" ? (e.target.value ? Number(e.target.value) : null) : e.target.value)
        }
      />
    </div>
  );

  return (
    <div className="grid grid-cols-2 gap-3 text-sm max-h-[65vh] overflow-y-auto pr-1">
      {field("Plate Number *", "plate_number", "text", "e.g. MH04CE8821")}
      <div className="space-y-1">
        <Label className="text-xs text-muted-foreground">Vehicle Type</Label>
        <Select
          value={values.vehicle_type ?? "Car"}
          onValueChange={(v) => set("vehicle_type", v)}
        >
          <SelectTrigger className="h-8 text-sm bg-background">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {VEHICLE_TYPES.map((t) => (
              <SelectItem key={t} value={t}>{t}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {field("Make *", "vehicle_make", "text", "e.g. Maruti Suzuki")}
      {field("Model *", "vehicle_model", "text", "e.g. Swift")}
      {field("Color *", "vehicle_color", "text", "e.g. White")}
      {field("Year", "vehicle_year", "number", "e.g. 2022")}
      {field("Owner Name *", "owner_name", "text")}
      {field("Owner Phone", "owner_phone", "tel")}
      {field("Owner Email", "owner_email", "email")}
      <div className="col-span-2 space-y-1">
        <Label className="text-xs text-muted-foreground">Owner Address</Label>
        <Textarea
          placeholder="Full address"
          value={values.owner_address ?? ""}
          rows={2}
          className="text-sm bg-background resize-none"
          onChange={(e) => set("owner_address", e.target.value)}
        />
      </div>
      {field("RTO State *", "registered_rto_state", "text", "e.g. Maharashtra")}
      {field("RTO Code *", "registered_rto_code", "text", "e.g. MH04")}
      {field("Chassis Number", "chassis_number")}
      {field("Engine Number", "engine_number")}
      {field("Reg. Date", "registration_date", "date")}
      {field("Reg. Expiry", "registration_expiry", "date")}
      {field("Insurance Expiry", "insurance_expiry", "date")}
    </div>
  );
}

// ─── main page ────────────────────────────────────────────────────────────────

export default function VehicleDatabase() {
  const { toast } = useToast();
  const [vehicles, setVehicles] = useState<VehicleRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("All");

  // modal state machine
  type ModalKind = "none" | "add" | "edit" | "complaint" | "recover" | "delete";
  const [modal, setModal] = useState<ModalKind>("none");
  const [target, setTarget] = useState<VehicleRecord | null>(null);
  const [saving, setSaving] = useState(false);

  // form states
  const [addForm, setAddForm] = useState<VehicleCreate>(emptyCreate());
  const [editForm, setEditForm] = useState<VehicleCreate>(emptyCreate());
  const [complaintForm, setComplaintForm] = useState<ComplaintRequest>({
    complaint_id: "", reported_by: "", reporting_station: "", theft_location: "",
  });
  const [recoverForm, setRecoverForm] = useState<RecoveryRequest>({
    resolution_notes: "", officer: "",
  });

  // ── load ────────────────────────────────────────────────────────────────────

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listVehicles({
        status: statusFilter === "All" ? "" : statusFilter,
        search,
      });
      setVehicles(data);
    } catch (e: unknown) {
      toast({ title: "Error loading vehicles", description: String(e), variant: "destructive" });
    } finally {
      setLoading(false);
    }
  }, [search, statusFilter, toast]);

  useEffect(() => { load(); }, [load]);

  // ── open modals ─────────────────────────────────────────────────────────────

  const openEdit = (v: VehicleRecord) => {
    setTarget(v);
    setEditForm({
      plate_number: v.plate_number,
      vehicle_make: v.vehicle_make,
      vehicle_model: v.vehicle_model,
      vehicle_year: v.vehicle_year,
      vehicle_color: v.vehicle_color,
      vehicle_type: v.vehicle_type,
      owner_name: v.owner_name,
      owner_phone: v.owner_phone ?? "",
      owner_email: v.owner_email ?? "",
      owner_address: v.owner_address ?? "",
      registered_rto_state: v.registered_rto_state,
      registered_rto_code: v.registered_rto_code,
      chassis_number: v.chassis_number ?? "",
      engine_number: v.engine_number ?? "",
      registration_date: v.registration_date ?? "",
      registration_expiry: v.registration_expiry ?? "",
      insurance_expiry: v.insurance_expiry ?? "",
    });
    setModal("edit");
  };

  const openComplaint = (v: VehicleRecord) => {
    setTarget(v);
    setComplaintForm({ complaint_id: "", reported_by: "", reporting_station: "", theft_location: "" });
    setModal("complaint");
  };

  const openRecover = (v: VehicleRecord) => {
    setTarget(v);
    setRecoverForm({ resolution_notes: "", officer: "" });
    setModal("recover");
  };

  const openDelete = (v: VehicleRecord) => {
    setTarget(v);
    setModal("delete");
  };

  const closeModal = () => { setModal("none"); setTarget(null); setSaving(false); };

  // ── mutations ────────────────────────────────────────────────────────────────

  const handleAdd = async () => {
    const required: (keyof VehicleCreate)[] = [
      "plate_number", "vehicle_make", "vehicle_model", "vehicle_color",
      "owner_name", "registered_rto_state", "registered_rto_code",
    ];
    const missing = required.filter((k) => !addForm[k]);
    if (missing.length) {
      toast({ title: "Missing required fields", description: missing.join(", "), variant: "destructive" });
      return;
    }
    setSaving(true);
    try {
      await createVehicle(addForm);
      toast({ title: "Vehicle added", description: addForm.plate_number });
      setAddForm(emptyCreate());
      closeModal();
      load();
    } catch (e: unknown) {
      toast({ title: "Failed to add vehicle", description: String(e), variant: "destructive" });
      setSaving(false);
    }
  };

  const handleEdit = async () => {
    if (!target) return;
    setSaving(true);
    const payload: VehicleUpdate = { ...editForm };
    delete (payload as Record<string, unknown>)["plate_number"];
    try {
      await updateVehicle(target.plate_number, payload);
      toast({ title: "Vehicle updated", description: target.plate_number });
      closeModal();
      load();
    } catch (e: unknown) {
      toast({ title: "Update failed", description: String(e), variant: "destructive" });
      setSaving(false);
    }
  };

  const handleComplaint = async () => {
    if (!target || !complaintForm.complaint_id.trim()) {
      toast({ title: "FIR/Complaint ID is required", variant: "destructive" });
      return;
    }
    setSaving(true);
    try {
      await fileComplaint(target.plate_number, complaintForm);
      toast({ title: "Complaint filed", description: `FIR ${complaintForm.complaint_id} for ${target.plate_number}` });
      closeModal();
      load();
    } catch (e: unknown) {
      toast({ title: "Failed to file complaint", description: String(e), variant: "destructive" });
      setSaving(false);
    }
  };

  const handleRecover = async () => {
    if (!target) return;
    setSaving(true);
    try {
      await markRecovered(target.plate_number, recoverForm);
      toast({ title: "Marked as recovered", description: target.plate_number });
      closeModal();
      load();
    } catch (e: unknown) {
      toast({ title: "Failed to update", description: String(e), variant: "destructive" });
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!target) return;
    setSaving(true);
    try {
      await deleteVehicle(target.plate_number);
      toast({ title: "Vehicle deleted", description: target.plate_number });
      closeModal();
      load();
    } catch (e: unknown) {
      toast({ title: "Delete failed", description: String(e), variant: "destructive" });
      setSaving(false);
    }
  };

  // ── counts ───────────────────────────────────────────────────────────────────

  const counts = {
    total: vehicles.length,
    clear: vehicles.filter((v) => v.status === "Clear").length,
    stolen: vehicles.filter((v) => v.status === "Stolen/Missing").length,
    recovered: vehicles.filter((v) => v.status === "Recovered").length,
  };

  // ── render ───────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Navbar */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass border-b border-border">
        <div className="container mx-auto flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-4">
            <Link to="/" className="flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors text-sm">
              <ArrowLeft className="h-4 w-4" /> Home
            </Link>
            <span className="text-border">|</span>
            <div className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-primary" />
              <span className="text-sm font-bold tracking-widest uppercase">Vehicle Registry</span>
            </div>
          </div>
          <Button size="sm" onClick={() => { setAddForm(emptyCreate()); setModal("add"); }}
            className="gap-1 text-xs">
            <Plus className="h-3.5 w-3.5" /> Add Vehicle
          </Button>
        </div>
      </nav>

      <div className="container mx-auto px-6 pt-24 pb-12">
        {/* Stat row */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
          {[
            { label: "Total Vehicles", value: counts.total, cls: "text-foreground" },
            { label: "Clear", value: counts.clear, cls: "text-green-400" },
            { label: "Stolen / Missing", value: counts.stolen, cls: "text-red-400" },
            { label: "Recovered", value: counts.recovered, cls: "text-yellow-400" },
          ].map((s) => (
            <div key={s.label}
              className="glass rounded-xl p-4 border border-border text-center">
              <div className={`text-2xl font-bold ${s.cls}`}>{s.value}</div>
              <div className="text-xs text-muted-foreground mt-1">{s.label}</div>
            </div>
          ))}
        </div>

        {/* Search + filter */}
        <div className="flex flex-col sm:flex-row gap-3 mb-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search by plate, owner, make, state…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9 bg-background"
            />
          </div>
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-44 bg-background">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {["All", "Clear", "Stolen/Missing", "Recovered"].map((s) => (
                <SelectItem key={s} value={s}>{s}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline" size="icon" onClick={load} title="Refresh">
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          </Button>
        </div>

        {/* Table */}
        <div className="rounded-xl border border-border overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-muted/40">
                  {["Plate", "Make / Model", "Color / Type", "Owner", "RTO State", "Status", "Actions"].map((h) => (
                    <th key={h} className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <AnimatePresence mode="popLayout">
                  {loading ? (
                    <tr>
                      <td colSpan={7} className="px-4 py-12 text-center text-muted-foreground">
                        <RefreshCw className="h-5 w-5 animate-spin mx-auto mb-2" />
                        Loading vehicles…
                      </td>
                    </tr>
                  ) : vehicles.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="px-4 py-12 text-center text-muted-foreground">
                        No vehicles found.
                      </td>
                    </tr>
                  ) : (
                    vehicles.map((v) => (
                      <motion.tr
                        key={v.plate_number}
                        initial={{ opacity: 0, y: 4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className="border-b border-border hover:bg-muted/30 transition-colors"
                      >
                        <td className="px-4 py-3 font-mono font-semibold tracking-wider text-primary">
                          {v.plate_number}
                        </td>
                        <td className="px-4 py-3">
                          <div>{v.vehicle_make} {v.vehicle_model}</div>
                          <div className="text-xs text-muted-foreground">{v.vehicle_year ?? "—"}</div>
                        </td>
                        <td className="px-4 py-3">
                          <div>{v.vehicle_color}</div>
                          <div className="text-xs text-muted-foreground">{v.vehicle_type}</div>
                        </td>
                        <td className="px-4 py-3">
                          <div>{v.owner_name}</div>
                          <div className="text-xs text-muted-foreground">{v.owner_phone ?? "—"}</div>
                        </td>
                        <td className="px-4 py-3">{v.registered_rto_state}</td>
                        <td className="px-4 py-3">{statusBadge(v.status)}</td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-1 flex-wrap">
                            <Button size="icon" variant="ghost" className="h-7 w-7" title="Edit"
                              onClick={() => openEdit(v)}>
                              <Edit2 className="h-3.5 w-3.5" />
                            </Button>
                            {v.status !== "Stolen/Missing" && (
                              <Button size="sm" variant="ghost"
                                className="h-7 px-2 text-xs text-red-400 hover:text-red-300"
                                title="File Complaint"
                                onClick={() => openComplaint(v)}>
                                <AlertTriangle className="h-3 w-3 mr-1" /> FIR
                              </Button>
                            )}
                            {v.status === "Stolen/Missing" && (
                              <Button size="sm" variant="ghost"
                                className="h-7 px-2 text-xs text-yellow-400 hover:text-yellow-300"
                                title="Mark Recovered"
                                onClick={() => openRecover(v)}>
                                <CheckCircle2 className="h-3 w-3 mr-1" /> Recovered
                              </Button>
                            )}
                            <Button size="icon" variant="ghost"
                              className="h-7 w-7 text-red-500 hover:text-red-400"
                              title="Delete"
                              onClick={() => openDelete(v)}>
                              <Trash2 className="h-3.5 w-3.5" />
                            </Button>
                          </div>
                        </td>
                      </motion.tr>
                    ))
                  )}
                </AnimatePresence>
              </tbody>
            </table>
          </div>
        </div>
        <p className="text-xs text-muted-foreground mt-3 text-right">
          {vehicles.length} record{vehicles.length !== 1 ? "s" : ""} shown
        </p>
      </div>

      {/* ── Add Vehicle Modal ──────────────────────────────────────────────── */}
      <Dialog open={modal === "add"} onOpenChange={(o) => { if (!o) closeModal(); }}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Add Vehicle</DialogTitle>
          </DialogHeader>
          <VehicleForm values={addForm} onChange={setAddForm} />
          <DialogFooter>
            <Button variant="outline" onClick={closeModal} disabled={saving}>Cancel</Button>
            <Button onClick={handleAdd} disabled={saving}>
              {saving ? "Adding…" : "Add Vehicle"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* ── Edit Vehicle Modal ─────────────────────────────────────────────── */}
      <Dialog open={modal === "edit"} onOpenChange={(o) => { if (!o) closeModal(); }}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Edit Vehicle — {target?.plate_number}</DialogTitle>
          </DialogHeader>
          <VehicleForm values={editForm} onChange={setEditForm} readonlyPlate />
          <DialogFooter>
            <Button variant="outline" onClick={closeModal} disabled={saving}>Cancel</Button>
            <Button onClick={handleEdit} disabled={saving}>
              {saving ? "Saving…" : "Save Changes"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* ── File Complaint Modal ───────────────────────────────────────────── */}
      <Dialog open={modal === "complaint"} onOpenChange={(o) => { if (!o) closeModal(); }}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle className="text-red-400 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" /> File Police Complaint
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-3 text-sm">
            <p className="text-muted-foreground text-xs">
              Vehicle: <span className="font-mono text-foreground">{target?.plate_number}</span>
            </p>
            {(
              [
                ["FIR / Complaint ID *", "complaint_id", "e.g. FIR/2024/MH04/001"],
                ["Reported By", "reported_by", "Name of individual filing FIR"],
                ["Reporting Station", "reporting_station", "e.g. Andheri Police Station"],
                ["Theft Location", "theft_location", "Where was the vehicle last seen?"],
              ] as [string, keyof ComplaintRequest, string][]
            ).map(([label, key, ph]) => (
              <div key={key} className="space-y-1">
                <Label className="text-xs text-muted-foreground">{label}</Label>
                <Input
                  placeholder={ph}
                  value={complaintForm[key] ?? ""}
                  className="h-8 text-sm bg-background"
                  onChange={(e) => setComplaintForm((p) => ({ ...p, [key]: e.target.value }))}
                />
              </div>
            ))}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={closeModal} disabled={saving}>Cancel</Button>
            <Button variant="destructive" onClick={handleComplaint} disabled={saving}>
              {saving ? "Filing…" : "File Complaint"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* ── Mark Recovered Modal ──────────────────────────────────────────── */}
      <Dialog open={modal === "recover"} onOpenChange={(o) => { if (!o) closeModal(); }}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle className="text-yellow-400 flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4" /> Mark as Recovered
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-3 text-sm">
            <p className="text-muted-foreground text-xs">
              Vehicle: <span className="font-mono text-foreground">{target?.plate_number}</span>
              {target?.police_complaint_id && (
                <span className="ml-2 text-muted-foreground">FIR: {target.police_complaint_id}</span>
              )}
            </p>
            <div className="space-y-1">
              <Label className="text-xs text-muted-foreground">Recovering Officer</Label>
              <Input
                placeholder="Officer name / badge number"
                value={recoverForm.officer ?? ""}
                className="h-8 text-sm bg-background"
                onChange={(e) => setRecoverForm((p) => ({ ...p, officer: e.target.value }))}
              />
            </div>
            <div className="space-y-1">
              <Label className="text-xs text-muted-foreground">Resolution Notes</Label>
              <Textarea
                placeholder="Where / how was the vehicle found?"
                rows={3}
                value={recoverForm.resolution_notes ?? ""}
                className="text-sm bg-background resize-none"
                onChange={(e) => setRecoverForm((p) => ({ ...p, resolution_notes: e.target.value }))}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={closeModal} disabled={saving}>Cancel</Button>
            <Button
              className="bg-yellow-500 text-black hover:bg-yellow-400"
              onClick={handleRecover}
              disabled={saving}
            >
              {saving ? "Updating…" : "Mark Recovered"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* ── Delete Confirmation Modal ─────────────────────────────────────── */}
      <Dialog open={modal === "delete"} onOpenChange={(o) => { if (!o) closeModal(); }}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle className="text-red-400 flex items-center gap-2">
              <Trash2 className="h-4 w-4" /> Delete Vehicle
            </DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">
            Are you sure you want to delete{" "}
            <span className="font-mono text-foreground">{target?.plate_number}</span>?
            This cannot be undone.
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={closeModal} disabled={saving}>Cancel</Button>
            <Button variant="destructive" onClick={handleDelete} disabled={saving}>
              {saving ? "Deleting…" : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
