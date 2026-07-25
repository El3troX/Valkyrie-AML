'use client';

import React, { useEffect, useState } from 'react';
import { api, AnomalyItem } from '@/lib/api';

export const Ticker: React.FC = () => {
  const [alerts, setAlerts] = useState<AnomalyItem[]>([]);

  useEffect(() => {
    // Fetch top 5 anomalies for ticker
    api.getTopAnomalies(5, 0.5)
      .then((data) => {
        setAlerts(data.anomalies);
      })
      .catch((err) => {
        console.error('Failed to load ticker alerts', err);
      });
  }, []);

  return (
    <div className="ticker-wrap h-[34px] flex items-center border-b-3 border-[#0A0A0A] bg-[#0A0A0A] text-[#F2F0EB]" aria-hidden="true">
      <div className="ticker-track">
        {alerts.length > 0 ? (
          // Duplicate alerts for infinite scrolling effect
          [...alerts, ...alerts, ...alerts].map((alert, i) => (
            <div key={i} className="ticker-item font-mono text-[10px] uppercase font-bold tracking-wider flex items-center gap-2">
              <span className="h-1.5 w-1.5 bg-[#E63946] animate-pulse" />
              <span>[ALERT]</span>
              <span>SENDER: <b>{alert.sender.slice(0, 10)}...</b></span>
              <span>AMOUNT: <b>${alert.amount.toLocaleString()}</b></span>
              <span>RISK: <b className="text-[#E63946]">{alert.risk_level}</b></span>
              <span>TYPOLOGY: <b className="text-[#D4A843]">{alert.laundering_type}</b></span>
              <span className="text-[#6b6f76]">◆</span>
            </div>
          ))
        ) : (
          <div className="ticker-track">
            <div className="ticker-item font-mono text-[10px] uppercase font-bold tracking-wider">
              ◆ 🛡️ VALKYRIE AML AGENT ACTIVE ◆ MONITORING 200,000 LIVE TRANSACTION FEEDS ◆ SHAP COMPLIANCE EXPLAINER STATUS: OK ◆ personalize pagerank propagation: active ◆
            </div>
            <div className="ticker-item font-mono text-[10px] uppercase font-bold tracking-wider">
              ◆ 🛡️ VALKYRIE AML AGENT ACTIVE ◆ MONITORING 200,000 LIVE TRANSACTION FEEDS ◆ SHAP COMPLIANCE EXPLAINER STATUS: OK ◆ personalize pagerank propagation: active ◆
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
