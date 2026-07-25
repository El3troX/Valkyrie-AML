'use client';

import React, { useState, useEffect } from 'react';
import { api, ModelPerformance } from '@/lib/api';
import { NeuCard } from '../shared/NeuCard';
import { NeuBtn } from '../shared/NeuBtn';
import { formatPercentage } from '@/lib/utils';
import { Target, AlertTriangle, CheckSquare } from 'lucide-react';

interface PerformancePanelProps {
  activeThreshold: number;
  onThresholdChange: (t: number) => void;
}

export const PerformancePanel: React.FC<PerformancePanelProps> = ({
  activeThreshold,
  onThresholdChange,
}) => {
  const [perfData, setPerfData] = useState<ModelPerformance | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    api.getModelPerformance(activeThreshold)
      .then((data) => {
        setPerfData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load performance metrics', err);
        setLoading(false);
      });
  }, [activeThreshold]);

  const applyOptimal = () => {
    if (perfData) {
      onThresholdChange(perfData.optimal_threshold);
    }
  };

  if (!perfData) {
    return (
      <div className="flex items-center justify-center p-12">
        <span className="font-mono text-xs uppercase animate-pulse">Loading model performance parameters...</span>
      </div>
    );
  }

  const o = perfData;

  return (
    <div className="flex flex-col gap-6">
      {/* Auto-tune CTA banner */}
      <div className="flex flex-col md:flex-row items-center justify-between gap-4 border-3 border-black bg-white p-4 [box-shadow:4px_4px_0_#0A0A0A]">
        <div>
          <h4 className="font-display font-extrabold text-sm uppercase">Auto-Tune Detection Threshold</h4>
          <p className="font-mono text-[10px] text-[#6b6f76] mt-1">
            Sweep isolation forest contamination ranges to optimize model F1-score.
          </p>
        </div>
        <NeuBtn variant="ink" onClick={applyOptimal} className="min-h-[40px] px-4 py-2 text-xs">
          <Target className="h-4 w-4" />
          Apply Optimal ({o.optimal_threshold.toFixed(4)})
        </NeuBtn>
      </div>

      {/* KPI Metric cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="border-3 border-black p-4 bg-white [box-shadow:3px_3px_0_#0A0A0A]">
          <div className="font-display font-extrabold text-3xl text-[#2EC04A]">
            {formatPercentage(o.precision)}
          </div>
          <div className="font-mono text-[9px] uppercase tracking-wider text-[#6b6f76] mt-1">
            Precision
          </div>
        </div>

        <div className="border-3 border-black p-4 bg-white [box-shadow:3px_3px_0_#0A0A0A]">
          <div className="font-display font-extrabold text-3xl text-[#5BC0EB]">
            {formatPercentage(o.recall)}
          </div>
          <div className="font-mono text-[9px] uppercase tracking-wider text-[#6b6f76] mt-1">
            Recall
          </div>
        </div>

        <div className="border-3 border-black p-4 bg-white [box-shadow:3px_3px_0_#0A0A0A]">
          <div className="font-display font-extrabold text-3xl text-[#D4A843]">
            {formatPercentage(o.f1)}
          </div>
          <div className="font-mono text-[9px] uppercase tracking-wider text-[#6b6f76] mt-1">
            F1 Score
          </div>
        </div>

        <div className="border-3 border-black p-4 bg-white [box-shadow:3px_3px_0_#0A0A0A]">
          <div className="font-display font-extrabold text-3xl text-[#E63946]">
            {o.false_positives.toLocaleString()}
          </div>
          <div className="font-mono text-[9px] uppercase tracking-wider text-[#6b6f76] mt-1">
            False Positives
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confusion Matrix */}
        <NeuCard>
          <h4 className="font-display font-extrabold text-sm mb-4">Confusion Matrix</h4>
          <div className="border-2 border-black overflow-hidden font-mono text-xs">
            <div className="grid grid-cols-3 bg-[#F2F0EB] font-bold text-center border-b-2 border-black">
              <div className="p-3 text-left border-r-2 border-black font-display text-[9px] tracking-wider">ACTUAL / PREDICTED</div>
              <div className="p-3 border-r-2 border-black">PRED NORMAL</div>
              <div className="p-3">PRED SUSPICIOUS</div>
            </div>
            <div className="grid grid-cols-3 border-b-2 border-black last:border-0 text-center">
              <div className="p-3 text-left font-bold bg-[#F2F0EB] border-r-2 border-black">ACTUAL NORMAL</div>
              <div className="p-3 border-r-2 border-black text-[#6b6f76]">{o.true_negatives.toLocaleString()}</div>
              <div className="p-3 text-[#E63946] font-bold">{o.false_positives.toLocaleString()}</div>
            </div>
            <div className="grid grid-cols-3 text-center">
              <div className="p-3 text-left font-bold bg-[#F2F0EB] border-r-2 border-black">ACTUAL SUSPICIOUS</div>
              <div className="p-3 border-r-2 border-black text-[#F97316] font-bold">{o.false_negatives.toLocaleString()}</div>
              <div className="p-3 text-[#2EC04A] font-bold">{o.true_positives.toLocaleString()}</div>
            </div>
          </div>
        </NeuCard>

        {/* Per-Typology performance */}
        <NeuCard>
          <h4 className="font-display font-extrabold text-sm mb-4">Per-Typology Breakdown</h4>
          <div className="overflow-x-auto">
            <table className="neu-table">
              <thead>
                <tr>
                  <th>Typology</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>Support</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(o.by_typology || {}).map(([typology, metrics]) => (
                  <tr key={typology}>
                    <td className="font-bold capitalize">{typology}</td>
                    <td>{formatPercentage(metrics.precision)}</td>
                    <td>{formatPercentage(metrics.recall)}</td>
                    <td className="font-bold text-[#D4A843]">{formatPercentage(metrics.f1)}</td>
                    <td className="text-[#6b6f76]">{metrics.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </NeuCard>
      </div>

      {/* Illustrative true positive list */}
      <NeuCard>
        <h4 className="font-display font-extrabold text-sm mb-4">Top True Positive Illustrative Cases</h4>
        <div className="overflow-x-auto">
          <table className="neu-table">
            <thead>
              <tr>
                <th>Sender Account</th>
                <th>Receiver Account</th>
                <th>Amount</th>
                <th>Typology Pattern</th>
                <th>Anomaly Score</th>
              </tr>
            </thead>
            <tbody>
              {o.illustrative_cases?.map((c, i) => (
                <tr key={i}>
                  <td className="font-bold">{c.Sender_account}</td>
                  <td>{c.Receiver_account}</td>
                  <td>${c.Amount?.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                  <td className="text-[#E63946] font-bold uppercase">{c.Laundering_type}</td>
                  <td className="font-bold text-[#D4A843]">{c.anomaly_score?.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </NeuCard>
    </div>
  );
};
