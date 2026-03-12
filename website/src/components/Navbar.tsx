import { useState } from "react";
import { Shield, Menu, X, Database } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Link } from "react-router-dom";

const NAV_ITEMS = [
  { label: "Model Demo", id: "pipeline" },
  { label: "Dashboard", id: "dashboard" },
];

const Navbar = () => {
  const [mobileOpen, setMobileOpen] = useState(false);

  const scrollTo = (id: string) => {
    setMobileOpen(false);
    // Small delay so the menu closes before scrolling (avoids layout jump)
    setTimeout(() => {
      document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
    }, 150);
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

        {/* Desktop links */}
        <div className="hidden md:flex items-center gap-8">
          {NAV_ITEMS.map((item) => (
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

        {/* Mobile hamburger */}
        <button
          className="md:hidden text-foreground p-1"
          onClick={() => setMobileOpen((o) => !o)}
          aria-label={mobileOpen ? "Close menu" : "Open menu"}
        >
          {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {/* Mobile dropdown panel */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="md:hidden overflow-hidden border-t border-border"
          >
            <div className="container mx-auto px-6 py-4 flex flex-col gap-4">
              {NAV_ITEMS.map((item) => (
                <button
                  key={item.id}
                  onClick={() => scrollTo(item.id)}
                  className="text-sm uppercase tracking-widest-xl text-muted-foreground hover:text-foreground transition-colors text-left"
                >
                  {item.label}
                </button>
              ))}
              <Link
                to="/vehicles"
                onClick={() => setMobileOpen(false)}
                className="flex items-center gap-1.5 text-sm uppercase tracking-widest-xl text-muted-foreground hover:text-foreground transition-colors"
              >
                <Database className="h-3.5 w-3.5" />
                Vehicle DB
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  );
};

export default Navbar;
