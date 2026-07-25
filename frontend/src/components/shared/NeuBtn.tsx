import React from 'react';
import { cn } from '@/lib/utils';

interface NeuBtnProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  variant?: 'default' | 'critical' | 'warning' | 'success' | 'info' | 'ink';
}

export const NeuBtn: React.FC<NeuBtnProps> = ({
  children,
  className,
  variant = 'default',
  ...props
}) => {
  const getVariantStyles = () => {
    switch (variant) {
      case 'critical':
        return 'bg-[#E63946] text-white hover:bg-[#d62839]';
      case 'warning':
        return 'bg-[#D4A843] text-black hover:bg-[#c29632]';
      case 'success':
        return 'bg-[#2EC04A] text-black hover:bg-[#25a33b]';
      case 'info':
        return 'bg-[#5BC0EB] text-black hover:bg-[#48afd9]';
      case 'ink':
        return 'bg-[#0A0A0A] text-[#F2F0EB] hover:bg-[#1a1a24] [box-shadow:4px_4px_0_#D4A843] hover:[box-shadow:6px_6px_0_#D4A843]';
      default:
        return 'bg-white text-black hover:bg-[#f3f0e8]';
    }
  };

  return (
    <button
      className={cn(
        'inline-flex items-center justify-center gap-2 border-3 border-[#0A0A0A] font-bold text-xs uppercase tracking-wider',
        'px-6 py-3 min-h-[48px] cursor-pointer transition-all duration-120 ease-[cubic-bezier(0.2,0,0,1)]',
        'active:translate-x-[2px] active:translate-y-[2px] active:box-shadow-[2px_2px_0_#0A0A0A]',
        variant !== 'ink' && '[box-shadow:4px_4px_0_#0A0A0A] hover:-translate-x-[2px] hover:-translate-y-[2px] hover:[box-shadow:6px_6px_0_#0A0A0A]',
        getVariantStyles(),
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
};
