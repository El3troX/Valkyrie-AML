import React from 'react';
import { cn } from '@/lib/utils';

interface NeuCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  variant?: 'default' | 'critical' | 'warning' | 'success' | 'info' | 'dark';
  glow?: boolean;
}

export const NeuCard: React.FC<NeuCardProps> = ({
  children,
  className,
  variant = 'default',
  glow = false,
  ...props
}) => {
  const getVariantStyles = () => {
    switch (variant) {
      case 'critical':
        return 'border-[#0A0A0A] bg-white text-[#0A0A0A] [box-shadow:6px_6px_0_#E63946]';
      case 'warning':
        return 'border-[#0A0A0A] bg-white text-[#0A0A0A] [box-shadow:6px_6px_0_#D4A843]';
      case 'success':
        return 'border-[#0A0A0A] bg-white text-[#0A0A0A] [box-shadow:6px_6px_0_#2EC04A]';
      case 'info':
        return 'border-[#0A0A0A] bg-white text-[#0A0A0A] [box-shadow:6px_6px_0_#5BC0EB]';
      case 'dark':
        return 'border-[#F2F0EB] bg-[#0A0A0F] text-[#F2F0EB] [box-shadow:6px_6px_0_#D4A843]';
      default:
        return 'border-[#0A0A0A] bg-white text-[#0A0A0A] [box-shadow:6px_6px_0_#0A0A0A]';
    }
  };

  return (
    <div
      className={cn(
        'border-3 p-6 transition-all duration-150 ease-[cubic-bezier(0.2,0,0,1)]',
        'hover:translate-x-[-2px] hover:translate-y-[-2px]',
        getVariantStyles(),
        variant !== 'dark' && 'hover:[box-shadow:8px_8px_0_#0A0A0A]',
        variant === 'dark' && 'hover:[box-shadow:8px_8px_0_#D4A843]',
        glow && 'animate-pulse',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};
