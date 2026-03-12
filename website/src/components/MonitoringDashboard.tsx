import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Zap, Database, Brain, TrendingUp } from 'lucide-react';

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
      label: 'OCR Accuracy',
      value: '22.0%',
      description: 'Smart pipeline +3.1pp',
      color: 'text-green-500',
    },
    {
      icon: Zap,
      label: 'Processing Speed',
      value: '~0.3s',
      description: 'Per image on GPU',
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
                        EasyOCR + Smart postprocessing (22% accuracy, Indian format
                        validation)
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
