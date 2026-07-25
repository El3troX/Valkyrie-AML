'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { Ticker } from '@/components/shared/Ticker';
import { NeuCard } from '@/components/shared/NeuCard';
import { NeuBtn } from '@/components/shared/NeuBtn';
import { RiskBadge } from '@/components/shared/RiskBadge';
import { InvestigationTerminal } from '@/components/agent/InvestigationTerminal';
import { NetworkGraph2D } from '@/components/network/NetworkGraph2D';
import { PerformancePanel } from '@/components/dashboard/PerformancePanel';
import { api, DashboardStats, AnomalyItem, NetworkData } from '@/lib/api';
import { formatCurrency, formatPercentage } from '@/lib/utils';
import { 
  ShieldAlert, 
  Terminal, 
  Network, 
  BarChart3, 
  FileText, 
  ArrowRight, 
  Search, 
  Download, 
  Filter, 
  AlertTriangle,
  RotateCcw
} from 'lucide-react';

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<'investigate' | 'network' | 'performance' | 'sar'>('investigate');
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [anomaliesData, setAnomaliesData] = useState<AnomalyItem[]>([]);
  const [networkData, setNetworkData] = useState<NetworkData | null>(null);
  
  // Filters state
  const [threshold, setThreshold] = useState(0.5);
  const [focusAccount, setFocusAccount] = useState('');
  const [activeQuery, setActiveQuery] = useState('');
  const [triggerSearch, setTriggerSearch] = useState(false);

  // SAR state
  const [sarAccount, setSarAccount] = useState('');
  const [sarNarrative, setSarNarrative] = useState('');
  const [sarData, setSarData] = useState<any>(null);
  const [generatingSar, setGeneratingSar] = useState(false);

  useEffect(() => {
    loadDashboardData();
  }, [threshold]);

  const loadDashboardData = () => {
    api.getDashboardStats()
      .then((data) => setStats(data))
      .catch((err) => console.error('Error fetching dashboard stats', err));

    api.getTopAnomalies(15, threshold)
      .then((data) => setAnomaliesData(data.anomalies))
      .catch((err) => console.error('Error fetching top anomalies', err));

    api.getNetworkData(60)
      .then((data) => setNetworkData(data))
      .catch((err) => console.error('Error fetching network data', err));
  };

  const handleAgentComplete = (data: any) => {
    loadDashboardData();
    
    // If agent returned get_anomaly_scores result, parse it
    if (data.results?.get_anomaly_scores?.top_anomalies) {
      setAnomaliesData(data.results.get_anomaly_scores.top_anomalies);
    }
  };

  const handleNodeClick = (accountId: string) => {
    setFocusAccount(accountId);
    // Auto query account profile in terminal
    setActiveQuery(`Is customer ID ${accountId} suspicious?`);
    setTriggerSearch(true);
    setActiveTab('investigate');
    // Reset trigger search flag after a delay
    setTimeout(() => setTriggerSearch(false), 200);
  };

  const handleGenerateSAR = (e: React.FormEvent) => {
    e.preventDefault();
    if (!sarAccount.trim()) return;
    setGeneratingSar(true);
    setSarNarrative('');
    
    api.generateSAR(sarAccount)
      .then((data) => {
        setSarNarrative(data.narrative);
        setSarData(data);
        setGeneratingSar(false);
      })
      .catch((err) => {
        console.error('Error compiling SAR', err);
        setSarNarrative(`Failed to generate Suspicious Activity Report narrative for Account ${sarAccount}. User does not exist or API timeout.`);
        setGeneratingSar(false);
      });
  };

  const triggerFocusInvestigation = () => {
    if (focusAccount.trim()) {
      setActiveQuery(`Is customer ID ${focusAccount} suspicious?`);
      setTriggerSearch(true);
      setTimeout(() => setTriggerSearch(false), 200);
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-[#F2F0EB]">
      {/* Alert Banner */}
      <Ticker />

      {/* Navigation */}
      <nav className="bg-[#F2F0EB] border-b-3 border-[#0A0A0A]">
        <div className="max-w-[1220px] mx-auto px-6 h-[72px] flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3">
            <span className="h-10 w-10 bg-[#0A0A0A] text-[#D4A843] border-3 border-[#0A0A0A] font-display font-bold text-xl flex items-center justify-center [box-shadow:2.5px_2.5px_0_#D4A843]">
              V
            </span>
            <span className="font-display font-extrabold text-xl tracking-wider text-[#0A0A0A]">
              VALKYRIE
            </span>
          </Link>
          <div className="font-mono text-[9px] uppercase tracking-wider text-[#6b6f76]">
            Compliance Dashboard Pilot v2.0
          </div>
        </div>
      </nav>

      {/* Header Stats */}
      <div className="border-b-3 border-[#0A0A0A] bg-white py-6">
        <div className="max-w-[1220px] mx-auto px-6 grid grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="kpi-card">
            <div className="kpi-value text-[#E63946]">
              {stats ? stats.flagged_transactions.toLocaleString() : '...'}
            </div>
            <div className="kpi-label">Flagged Detections</div>
          </div>

          <div className="kpi-card">
            <div className="kpi-value text-[#2EC04A]">
              {stats ? `${(stats.f1_score * 100).toFixed(1)}%` : '...'}
            </div>
            <div className="kpi-label">Model F1 accuracy</div>
          </div>

          <div className="kpi-card">
            <div className="kpi-value text-[#5BC0EB]">
              {stats ? stats.total_transactions.toLocaleString() : '...'}
            </div>
            <div className="kpi-label">Total transactions</div>
          </div>

          <div className="kpi-card">
            <div className="kpi-value text-[#D4A843]">
              {stats ? stats.optimal_threshold.toFixed(4) : '...'}
            </div>
            <div className="kpi-label">Optimal F1 threshold</div>
          </div>
        </div>
      </div>

      {/* Workspace Grid */}
      <main className="max-w-[1220px] mx-auto px-6 py-10 grid grid-cols-1 lg:grid-cols-12 gap-8 flex-1 w-full">
        {/* Left Control Panel / Sidebar */}
        <aside className="lg:col-span-4 flex flex-col gap-6">
          {/* Query filter sidebar */}
          <NeuCard className="bg-white">
            <div className="flex items-center gap-2 border-b-2 border-[#0A0A0A] pb-3 mb-4">
              <Filter className="h-4 w-4" />
              <h3 className="font-display font-extrabold text-sm uppercase">Global Filters</h3>
            </div>

            <div className="flex flex-col gap-5 font-mono text-xs">
              <div className="flex flex-col gap-2">
                <div className="flex justify-between font-bold text-[10px] uppercase">
                  <span>Risk Score Threshold</span>
                  <span className="text-[#E63946]">{threshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="w-full accent-black border border-black h-1 bg-gray-200"
                />
              </div>

              <div className="flex flex-col gap-2">
                <label className="font-bold text-[10px] uppercase">Focus Account ID</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={focusAccount}
                    onChange={(e) => setFocusAccount(e.target.value)}
                    placeholder="e.g. 207936746"
                    className="flex-1 border-2 border-black bg-white px-3 py-2 font-mono text-xs focus:outline-none"
                  />
                  {focusAccount && (
                    <button 
                      type="button" 
                      onClick={() => setFocusAccount('')}
                      className="border-2 border-black bg-[#F2F0EB] px-2.5 flex items-center justify-center"
                    >
                      <RotateCcw className="h-3 w-3" />
                    </button>
                  )}
                </div>
              </div>

              <NeuBtn 
                onClick={triggerFocusInvestigation} 
                disabled={!focusAccount.trim()}
                variant="ink" 
                className="w-full min-h-[40px] text-xs py-2"
              >
                <Search className="h-3 w-3" />
                Investigate Focus Account
              </NeuBtn>
            </div>
          </NeuCard>

          {/* Suspicious list */}
          <NeuCard className="flex-1 flex flex-col min-h-[300px]">
            <div className="flex items-center justify-between border-b-2 border-[#0A0A0A] pb-3 mb-4">
              <div className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-[#E63946] animate-pulse" />
                <h3 className="font-display font-extrabold text-sm uppercase">Suspicious Alerts</h3>
              </div>
              <span className="font-mono text-[9px] bg-black text-[#F2F0EB] px-2 py-0.5 font-bold uppercase">
                {anomaliesData.length} alerts
              </span>
            </div>

            <div className="flex-1 overflow-y-auto max-h-[360px] flex flex-col gap-3 pr-1">
              {anomaliesData.length > 0 ? (
                anomaliesData.map((alert, i) => (
                  <div 
                    key={i} 
                    onClick={() => handleNodeClick(alert.sender)}
                    className="border-2 border-black p-3 bg-white hover:bg-black/5 cursor-pointer [box-shadow:2px_2px_0_#0A0A0A] transition-all hover:translate-x-[-1px] hover:translate-y-[-1px] hover:[box-shadow:3px_3px_0_#0A0A0A]"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-mono text-[10px] font-bold">
                        ACC #{alert.sender.slice(0, 10)}
                      </span>
                      <RiskBadge level={alert.risk_level} />
                    </div>
                    <div className="flex justify-between font-mono text-[10px] text-[#6b6f76]">
                      <span>Amount: <b className="text-black">${alert.amount.toLocaleString()}</b></span>
                      <span className="uppercase text-[#E63946] font-bold">{alert.laundering_type}</span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="h-full flex items-center justify-center flex-col text-center p-6 border-2 border-dashed border-[#6b6f76]/40">
                  <ShieldAlert className="h-8 w-8 text-[#6b6f76]/40 mb-2" />
                  <span className="font-mono text-[10px] text-[#6b6f76]">No suspicious alerts above this threshold. Try lowering it.</span>
                </div>
              )}
            </div>
          </NeuCard>
        </aside>

        {/* Right workspace panels */}
        <section className="lg:col-span-8 flex flex-col gap-6">
          {/* Neubrutalist Workspace Tabs */}
          <div className="flex flex-wrap gap-2.5">
            {[
              { id: 'investigate', label: 'Investigation terminal', icon: Terminal },
              { id: 'network', label: 'Network Graph', icon: Network },
              { id: 'performance', label: 'Model parameters', icon: BarChart3 },
              { id: 'sar', label: 'SAR pdf compiler', icon: FileText },
            ].map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`inline-flex items-center gap-2 border-3 border-black font-display font-extrabold text-xs uppercase tracking-wider px-5 py-3.5 transition-all duration-120 cursor-pointer ${
                    activeTab === tab.id
                      ? 'bg-[#D4A843] [box-shadow:4px_4px_0_#0A0A0A] -translate-x-1 -translate-y-1'
                      : 'bg-white text-black hover:-translate-x-[2px] hover:-translate-y-[2px] hover:[box-shadow:4px_4px_0_#0A0A0A]'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* Active Workspace Viewport */}
          <div className="flex-1 bg-white border-3 border-black p-6 [box-shadow:6px_6px_0_#0A0A0A]">
            {activeTab === 'investigate' && (
              <InvestigationTerminal 
                onInvestigationComplete={handleAgentComplete} 
                activeQuery={activeQuery}
                triggerSearch={triggerSearch}
              />
            )}

            {activeTab === 'network' && (
              <div className="flex flex-col gap-4">
                <div className="border-b-2 border-black pb-3 flex items-start justify-between">
                  <div>
                    <h4 className="font-display font-extrabold text-sm uppercase">Personalised PageRank — Transaction Risk Graph</h4>
                    <p className="font-mono text-[10px] text-[#6b6f76] mt-1">
                      Force-directed 2D graph. Node size = PageRank centrality. Edge labels = aggregated $ flow. Click a node to investigate.
                    </p>
                  </div>
                  <button
                    onClick={() => api.getNetworkData(60).then(setNetworkData)}
                    className="font-mono text-[9px] uppercase tracking-wider border-2 border-black px-3 py-1.5 bg-[#F2F0EB] hover:bg-[#D4A843] transition-colors [box-shadow:2px_2px_0_#0A0A0A] flex-shrink-0 ml-4"
                  >
                    ↺ Refresh
                  </button>
                </div>
                <div className="relative border-2 border-black bg-[#0A0A0F] h-[520px]">
                  {networkData ? (
                    <NetworkGraph2D data={networkData} onNodeClick={handleNodeClick} />
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center font-mono text-xs uppercase tracking-widest text-[#D4A843] animate-pulse">
                      Building graph...
                    </div>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'performance' && (
              <PerformancePanel 
                activeThreshold={threshold} 
                onThresholdChange={(t) => setThreshold(t)}
              />
            )}

            {activeTab === 'sar' && (
              <div className="flex flex-col gap-6">
                <div className="border-b-2 border-black pb-3">
                  <h4 className="font-display font-extrabold text-sm uppercase">Suspicious Activity Report PDF Compiler</h4>
                  <p className="font-mono text-[10px] text-[#6b6f76] mt-1">
                    Enter flagged customer accounts to compile automated regulatory compliance reports.
                  </p>
                </div>

                <form onSubmit={handleGenerateSAR} className="flex gap-4">
                  <input
                    type="text"
                    value={sarAccount}
                    onChange={(e) => setSarAccount(e.target.value)}
                    placeholder="Enter customer Account ID..."
                    className="flex-1 border-3 border-black bg-[#F2F0EB] px-4 py-3 font-mono text-sm focus:outline-none focus:ring-0 focus:border-black placeholder-black/40 [box-shadow:3px_3px_0_#0A0A0A]"
                    disabled={generatingSar}
                  />
                  <NeuBtn variant="ink" type="submit" disabled={generatingSar}>
                    <FileText className="h-4 w-4" />
                    Compile SAR
                  </NeuBtn>
                </form>

                {generatingSar && (
                  <div className="flex items-center justify-center p-12 border-2 border-dashed border-black/20 flex-col gap-2 animate-pulse">
                    <span className="h-5 w-5 border-2 border-t-transparent border-[#D4A843] rounded-full animate-spin" />
                    <span className="font-mono text-xs uppercase">Connecting to Grok-3 to compile narrative structure...</span>
                  </div>
                )}

                {sarNarrative && (
                  <div className="flex flex-col gap-6">
                    {/* Visual details */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 border-2 border-black p-4 bg-[#F2F0EB]">
                      <div>
                        <span className="font-display text-[9px] uppercase tracking-wider text-[#6b6f76]">Seeded Account</span>
                        <div className="font-mono text-xs font-bold mt-1">#{sarData?.account_id}</div>
                      </div>
                      <div>
                        <span className="font-display text-[9px] uppercase tracking-wider text-[#6b6f76]">Aggregated Volume</span>
                        <div className="font-mono text-xs font-bold mt-1">${sarData?.anomaly_data?.total_amount?.toLocaleString()}</div>
                      </div>
                      <div>
                        <span className="font-display text-[9px] uppercase tracking-wider text-[#6b6f76]">Calculated Action</span>
                        <div className="font-mono text-xs font-bold mt-1 text-[#E63946] uppercase">{sarData?.escalation}</div>
                      </div>
                    </div>

                    <NeuCard className="bg-[#0A0A0F] text-white font-semibold font-mono text-xs p-5 max-h-[500px] overflow-y-auto leading-relaxed border-[#F2F0EB]/30 whitespace-pre-wrap">
                      <div className="text-[#D4A843] font-bold border-b border-white/20 pb-2 mb-3 uppercase tracking-widest text-sm">SAR NARRATIVE DRAFT (COMPLIANCE POINTERS):</div>
                      {sarNarrative}
                    </NeuCard>

                    <NeuBtn variant="success" className="w-fit" onClick={() => window.print()}>
                      <Download className="h-4 w-4" />
                      Print Compliance File
                    </NeuBtn>
                  </div>
                )}
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
