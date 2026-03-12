import { motion } from "framer-motion";

interface StatCardProps {
  value: string;
  label: string;
  delay?: number;
}

const StatCard = ({ value, label, delay = 0 }: StatCardProps) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.6, delay }}
    className="glass rounded-xl px-5 py-4 border-glow"
  >
    <div className="text-2xl font-bold text-foreground">{value}</div>
    <div className="text-xs uppercase tracking-widest-xl text-muted-foreground mt-1">
      {label}
    </div>
  </motion.div>
);

export default StatCard;
