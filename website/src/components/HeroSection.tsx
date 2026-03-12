import { motion } from "framer-motion";
import { ArrowRight, Layers } from "lucide-react";
import { Button } from "@/components/ui/button";
import StatCard from "./StatCard";
import heroImage from "@/assets/hero-vehicle.jpg";

const HeroSection = () => {
  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="relative min-h-screen flex items-center overflow-hidden">
      {/* Background image */}
      <div className="absolute inset-0">
        <img
          src={heroImage}
          alt="Sleek vehicle in moody lighting"
          className="w-full h-full object-cover opacity-40"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-background via-background/80 to-background/40" />
        <div className="absolute inset-0 bg-gradient-to-r from-background/90 via-transparent to-background/60" />
      </div>

      <div className="container relative z-10 mx-auto px-6 pt-24">
        <div className="max-w-4xl">
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-sm uppercase tracking-widest-xl text-primary mb-6"
          >
            YOLOv8 + EasyOCR Deep Learning Pipeline
          </motion.p>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="text-5xl md:text-7xl lg:text-8xl font-bold leading-[0.95] tracking-tight text-foreground mb-6"
          >
            SMART CITY
            <br />
            SECURITY.
            <br />
            <span className="text-gradient-accent">BEYOND TOMORROW.</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
            className="text-lg text-muted-foreground max-w-xl mb-10"
          >
            Intelligent Vehicle Monitoring and Theft Detection powered by
            YOLOv8 and EasyOCR.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
            className="flex flex-wrap gap-4"
          >
            <Button
              size="lg"
              onClick={() => scrollTo("pipeline")}
              className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold uppercase tracking-widest-xl text-sm px-8 glow-orange"
            >
              Try the Model
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="lg"
              onClick={() => scrollTo("dashboard")}
              className="border-border text-foreground hover:bg-secondary font-semibold uppercase tracking-widest-xl text-sm px-8"
            >
              <Layers className="mr-2 h-4 w-4" />
              View Architecture
            </Button>
          </motion.div>
        </div>

        {/* Floating stat cards */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="absolute bottom-12 right-6 md:right-12 flex flex-col gap-3"
        >
          <StatCard value="99.48%" label="Detection mAP" delay={0.9} />
          <StatCard value="~10.5s" label="Processing Time" delay={1.0} />
          <StatCard value="95.5%" label="OCR Accuracy" delay={1.1} />
        </motion.div>
      </div>
    </section>
  );
};

export default HeroSection;
