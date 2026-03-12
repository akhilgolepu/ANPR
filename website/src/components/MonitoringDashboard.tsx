import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Zap, Database, Brain, TrendingUp } from 'lucide-react';

// ── Detection log simulation ─────────────────────────────────────────────────

type VehicleKind = 'Car' | 'Truck' | 'Bus' | 'Bike' | 'Auto';
interface LogEntry {
  id: number;
  timestamp: string;
  vehicleType: VehicleKind;
  plateNumber: string;
  confidence: number;
  status: 'Clear' | 'ALERT';
}

const POOL: Omit<LogEntry, 'id' | 'timestamp'>[] = [
  { vehicleType: 'Car',   plateNumber: 'TS 09 AB 1234', confidence: 94, status: 'Clear'  },
  { vehicleType: 'Truck', plateNumber: 'AP 28 CD 5678', confidence: 88, status: 'Clear'  },
  { vehicleType: 'Car',   plateNumber: 'KA 01 EF 9012', confidence: 91, status: 'ALERT'  },
  { vehicleType: 'Bus',   plateNumber: 'TN 07 GH 3456', confidence: 87, status: 'Clear'  },
  { vehicleType: 'Car',   plateNumber: 'MH 12 IJ 7890', confidence: 93, status: 'Clear'  },
  { vehicleType: 'Car',   plateNumber: 'DL 05 KL 2345', confidence: 96, status: 'Clear'  },
  { vehicleType: 'Truck', plateNumber: 'RJ 14 MN 6789', confidence: 85, status: 'ALERT'  },
  { vehicleType: 'Car',   plateNumber: 'GJ 06 OP 1234', confidence: 92, status: 'Clear'  },
  { vehicleType: 'Bike',  plateNumber: 'PB 10 QR 5678', confidence: 89, status: 'Clear'  },
  { vehicleType: 'Car',   plateNumber: 'UP 16 BT 4490', confidence: 90, status: 'ALERT'  },
  { vehicleType: 'Auto',  plateNumber: 'MH 04 CE 8821', confidence: 95, status: 'Clear'  },
  { vehicleType: 'Car',   plateNumber: 'KA 05 MN 3301', confidence: 97, status: 'Clear'  },
];

function nowTime(): string {
  return new Date().toTimeString().slice(0, 8);
}

function buildInitial(): LogEntry[] {
  const base = new Date();
  return [...POOL].slice(0, 8).map((e, i) => {
    const d = new Date(base.getTime() - (7 - i) * 13_000);
    return { ...e, id: i, timestamp: d.toTimeString().slice(0, 8) };
  });
}

function DetectionLog() {
  const [entries, setEntries] = useState<LogEntry[]>(buildInitial);
  const idRef = useRef(POOL.length);
  const poolIdx = useRef(8 % POOL.length);

  useEffect(() => {
    const interval = setInterval(() => {
      const next = POOL[poolIdx.current % POOL.length];
      poolIdx.current++;
      setEntries((prev) => [
        { ...next, id: idRef.current++, timestamp: nowTime() },
        ...prev.slice(0, 9),
      ]);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      viewport={{ once: true }}
      className="mb-16"
    >
      <Card className="glass border-glow">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Real-Time Detection Feed</CardTitle>
          <p className="text-xs text-muted-foreground">
            Real-time vehicle detection log. Simulated smart city security terminal output.
          </p>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-muted/30">
                  {['Timestamp', 'Vehicle Type', 'Plate Number', 'Confidence', 'Status'].map((h) => (
                    <th key={h} className="px-5 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {entries.map((row, i) => (
                  <motion.tr
                    key={row.id}
                    initial={i === 0 ? { opacity: 0, backgroundColor: 'rgba(249,115,22,0.08)' } : false}
                    animate={{ opacity: 1, backgroundColor: 'rgba(0,0,0,0)' }}
                    transition={{ duration: 1.2 }}
                    className="border-b border-border/50 hover:bg-muted/20 transition-colors"
                  >
                    <td className="px-5 py-3 font-mono text-xs text-muted-foreground">{row.timestamp}</td>
                    <td className="px-5 py-3 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-primary/60 inline-block" />
                      {row.vehicleType}
                    </td>
                    <td className="px-5 py-3 font-mono font-semibold tracking-widest text-foreground">
                      {row.plateNumber}
                    </td>
                    <td className="px-5 py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-24 h-1.5 bg-muted rounded-full overflow-hidden">
                          <div
                            className="h-full bg-primary rounded-full transition-all duration-700"
                            style={{ width: `${row.confidence}%` }}
                          />
                        </div>
                        <span className="text-xs text-muted-foreground tabular-nums w-8">{row.confidence}%</span>
                      </div>
                    </td>
                    <td className="px-5 py-3">
                      {row.status === 'Clear' ? (
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-green-500/15 text-green-400 border border-green-500/30">
                          ✓ Clear
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-red-500/15 text-red-400 border border-red-500/30 animate-pulse">
                          ⚠ ALERT
                        </span>
                      )}
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

const MonitoringDashboard = () => {
  const metrics = [
    {
      icon: Brain,
      label: 'Detection Model',
      value: 'YOLOv8s',
      description: '99.48% mAP accuracy',
      color: 'text-blue-500',
    },
    {
      icon: TrendingUp,
      label: 'OCR Engine',
      value: 'TrOCR',
      description: 'microsoft/trocr-base-printed · avg 84% conf',
      color: 'text-green-500',
    },
    {
      icon: Zap,
      label: 'Processing Speed',
      value: '~3.4s',
      description: 'Per image (TrOCR)',
      color: 'text-orange-500',
    },
    {
      icon: Database,
      label: 'Dataset',
      value: '4,802',
      description: 'License plate samples',
      color: 'text-purple-500',
    },
  ];

  return (
    <section id="dashboard" className="py-24 bg-background">
      <div className="container mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          <DetectionLog />

          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold mb-4">System Performance</h2>
            <p className="text-lg text-muted-foreground">
              Advanced architecture optimized for real-time ANPR
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {metrics.map((metric, index) => {
              const Icon = metric.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <Card className="glass border-glow h-full hover:border-primary/50 transition-colors">
                    <CardHeader className="pb-4">
                      <div className={`inline-block p-2 rounded-lg bg-secondary ${metric.color} mb-3`}>
                        <Icon className="w-6 h-6" />
                      </div>
                      <CardTitle className="text-lg">{metric.label}</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="text-3xl font-bold text-gradient-accent">
                        {metric.value}
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {metric.description}
                      </p>
                      <Badge variant="secondary" className="mt-2">
                        Production Ready
                      </Badge>
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}
          </div>

          {/* Architecture Overview */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            viewport={{ once: true }}
            className="mt-12"
          >
            <Card className="glass border-glow">
              <CardHeader>
                <CardTitle>Pipeline Architecture</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center gap-4">
                    <div className="flex-1">
                      <p className="font-semibold text-sm mb-1">Image/Video Input</p>
                      <p className="text-xs text-muted-foreground">
                        Upload single/multiple images or video files
                      </p>
                    </div>
                    <Badge variant="outline">Step 1</Badge>
                  </div>

                  <div className="h-px bg-border" />

                  <div className="flex items-center gap-4">
                    <div className="flex-1">
                      <p className="font-semibold text-sm mb-1">
                        Vehicle & License Plate Detection
                      </p>
                      <p className="text-xs text-muted-foreground">
                        YOLOv8s model (99.48% mAP) detects vehicle regions and plate
                        regions
                      </p>
                    </div>
                    <Badge variant="outline" className="text-blue-500">
                      Step 2
                    </Badge>
                  </div>

                  <div className="h-px bg-border" />

                  <div className="flex items-center gap-4">
                    <div className="flex-1">
                      <p className="font-semibold text-sm mb-1">
                        License Plate Character Recognition
                      </p>
                      <p className="text-xs text-muted-foreground">
                        TrOCR (microsoft/trocr-base-printed) + Phase-2 preprocessing:
                        CLAHE contrast enhancement &amp; bilateral denoising
                      </p>
                    </div>
                    <Badge variant="outline" className="text-green-500">
                      Step 3
                    </Badge>
                  </div>

                  <div className="h-px bg-border" />

                  <div className="flex items-center gap-4">
                    <div className="flex-1">
                      <p className="font-semibold text-sm mb-1">
                        Results Visualization & Export
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Interactive results with crops, confidence scores, and export
                        options
                      </p>
                    </div>
                    <Badge variant="outline" className="text-orange-500">
                      Step 4
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};

export default MonitoringDashboard;
