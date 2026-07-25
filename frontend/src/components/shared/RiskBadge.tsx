import React from 'react';
import { cn } from '@/lib/utils';

interface RiskBadgeProps {
  level: string;
  className?: string;
}

export const RiskBadge: React.FC<RiskBadgeProps> = ({ level, className }) => {
  const getBadgeStyles = () => {
    switch (level.toUpperCase()) {
      case 'CRITICAL':
        return 'bg-[#E63946] text-white';
      case 'HIGH':
        return 'bg-[#F97316] text-white';
      case 'MEDIUM':
        return 'bg-[#EAB308] text-black';
      case 'LOW':
        return 'bg-[#2EC04A] text-black';
      default:
        return 'bg-[#6b6f76] text-white';
    }
  };

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 border-2 border-[#0A0A0A] font-bold text-[9px] uppercase tracking-wider',
        'px-2 py-0.5 [box-shadow:1.5px_1.5px_0_#0A0A0A]',
        getBadgeStyles(),
        className
      )}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-white border border-black/20" />
      {level}
    </span>
  );
};
