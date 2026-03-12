import Navbar from "@/components/Navbar";

import HeroSection from "@/components/HeroSection";
import PipelineDemo from "@/components/PipelineDemo";
import MonitoringDashboard from "@/components/MonitoringDashboard";

const Index = () => (
  <div className="min-h-screen bg-background">
    <Navbar />
    <HeroSection />
    <PipelineDemo />
    <MonitoringDashboard />
    <footer className="py-12 border-t border-border">
      <div className="container mx-auto px-6 text-center">
        <p className="text-sm text-muted-foreground">
          Intelligent Vehicle Monitoring & Theft Detection — Portfolio Project
        </p>
      </div>
    </footer>
  </div>
);

export default Index;
