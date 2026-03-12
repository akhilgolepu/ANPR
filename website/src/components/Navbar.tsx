import { Shield, Menu, Database } from "lucide-react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";

const Navbar = () => {
  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <motion.nav
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6 }}
      className="fixed top-0 left-0 right-0 z-50 glass"
    >
      <div className="container mx-auto flex items-center justify-between px-6 py-4">
        <div className="flex items-center gap-2">
          <Shield className="h-6 w-6 text-primary" />
          <span className="text-lg font-bold tracking-widest-xl uppercase text-foreground">
            ANPR
          </span>
        </div>

        <div className="hidden md:flex items-center gap-8">
          {[
            { label: "Model Demo", id: "pipeline" },
            { label: "Dashboard", id: "dashboard" },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => scrollTo(item.id)}
              className="text-sm uppercase tracking-widest-xl text-muted-foreground hover:text-foreground transition-colors"
            >
              {item.label}
            </button>
          ))}
          <Link
            to="/vehicles"
            className="flex items-center gap-1.5 text-sm uppercase tracking-widest-xl text-muted-foreground hover:text-foreground transition-colors"
          >
            <Database className="h-3.5 w-3.5" />
            Vehicle DB
          </Link>
        </div>

        <button className="md:hidden text-foreground">
          <Menu className="h-5 w-5" />
        </button>
      </div>
    </motion.nav>
  );
};

export default Navbar;
