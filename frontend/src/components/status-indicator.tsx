import { AlertCircle, CheckCircle2, Clock3, Loader2 } from "lucide-react";

type MediaStatus = "pending" | "processing" | "indexed" | "failed" | string;

type StatusIndicatorProps = {
  status?: MediaStatus | null;
  className?: string;
  showLabel?: boolean;
};

export function StatusIndicator({
  status,
  className = "",
  showLabel = false,
}: StatusIndicatorProps) {
  const normalized = status ?? "pending";
  const label = normalized.charAt(0).toUpperCase() + normalized.slice(1);

  const state = (() => {
    switch (normalized) {
      case "indexed":
        return {
          Icon: CheckCircle2,
          classes:
            "border-[rgba(17,255,153,0.26)] bg-[rgba(17,255,153,0.14)] text-[#7dffc7]",
        };
      case "processing":
        return {
          Icon: Loader2,
          classes:
            "border-[rgba(59,158,255,0.3)] bg-[rgba(0,129,253,0.18)] text-[#9fceff]",
          iconClass: "animate-spin",
        };
      case "failed":
        return {
          Icon: AlertCircle,
          classes:
            "border-[rgba(255,32,71,0.32)] bg-[rgba(255,32,71,0.16)] text-[#ff9bab]",
        };
      default:
        return {
          Icon: Clock3,
          classes:
            "border-[rgba(255,197,61,0.28)] bg-[rgba(255,197,61,0.14)] text-[#ffe08a]",
        };
    }
  })();

  const { Icon } = state;

  return (
    <span
      className={`inline-flex items-center justify-center gap-1.5 rounded-full border backdrop-blur-md ${state.classes} ${
        showLabel ? "px-2.5 py-1 text-xs font-medium" : "h-7 w-7"
      } ${className}`}
      title={label}
      role="img"
      aria-label={`Status: ${label}`}
    >
      <Icon className={`h-3.5 w-3.5 ${state.iconClass ?? ""}`} />
      {showLabel && <span>{label}</span>}
    </span>
  );
}
