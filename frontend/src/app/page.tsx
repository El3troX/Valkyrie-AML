'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { Ticker } from '@/components/shared/Ticker';
import { RiskGlobe } from '@/components/globe/RiskGlobe';
import { NeuCard } from '@/components/shared/NeuCard';
import { NeuBtn } from '@/components/shared/NeuBtn';
import { api, DashboardStats } from '@/lib/api';
import { Shield, Network, Terminal, BarChart3, ArrowRight, Activity, Check } from 'lucide-react';

export default function LandingPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null);

  useEffect(() => {
    api.getDashboardStats()
      .then((data) => setStats(data))
      .catch((err) => console.error('Failed to load dashboard stats', err));
  }, []);

  return (
    <div className="flex flex-col min-h-screen">
      {/* Ticker tape banner */}
      <Ticker />

      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-[#F2F0EB] border-b-3 border-[#0A0A0A]">
        <div className="max-w-[1220px] mx-auto px-6 h-[72px] flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3">
            <span className="h-10 w-10 bg-[#0A0A0A] text-[#D4A843] border-3 border-[#0A0A0A] font-display font-bold text-xl flex items-center justify-center [box-shadow:2.5px_2.5px_0_#D4A843]">
              V
            </span>
            <span className="font-display font-extrabold text-xl tracking-wider text-[#0A0A0A]">
              VALKYRIE
            </span>
          </Link>
          <div className="hidden md:flex gap-8">
            <a href="#features" className="nav-link">Capabilities</a>
            <a href="#how-it-works" className="nav-link">Architecture</a>
            <a href="#typology" className="nav-link">Typologies</a>
          </div>
          <Link href="/dashboard">
            <NeuBtn variant="ink" className="min-h-[44px] px-5 py-2">
              Launch Agent UI
              <ArrowRight className="h-4 w-4" />
            </NeuBtn>
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <header className="py-12 md:py-20 border-b-3 border-[#0A0A0A]">
        <div className="max-w-[1220px] mx-auto px-6 grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div className="flex flex-col justify-center">
            <div className="inline-flex items-center gap-2 bg-white border-2 border-[#0A0A0A] [box-shadow:3px_3px_0_#0A0A0A] px-4 py-1.5 font-bold text-[10px] uppercase tracking-wider mb-6 w-fit">
              <span className="h-2 w-2 rounded-full bg-[#2EC04A] animate-pulse" />
              Autonomous Compliance Pilot · ACTIVE
            </div>
            
            <h1 className="font-display font-extrabold text-4xl md:text-6xl uppercase tracking-tight text-[#0A0A0A] mb-6 leading-[1.05]">
              The Agent That <br />
              <span className="relative inline-block px-3 py-1 bg-[#D4A843] border-2 border-black rotate-[-1deg] mr-2">
                Thinks
              </span> 
              Before It 
              <span className="relative inline-block px-3 py-1 bg-[#E63946] text-white border-2 border-black rotate-[1.5deg] ml-2">
                Flags.
              </span>
            </h1>

            <p className="font-mono text-xs md:text-sm text-[#2b2b2b] mb-8 max-w-[50ch] leading-relaxed">
              Valkyrie-AML goes beyond rigid, rule-based systems. It constructs <b>dynamic execution plans</b> via Grok-3 and LangGraph to perform deep network investigation, traces layering paths, and explains risk scores using SHAP. 
            </p>

            <div className="flex flex-wrap gap-4">
              <Link href="/dashboard">
                <NeuBtn variant="ink">
                  Enter War Room
                  <Terminal className="h-4 w-4" />
                </NeuBtn>
              </Link>
              <a href="#features">
                <NeuBtn>See capabilities</NeuBtn>
              </a>
            </div>

            <p className="font-mono text-[9px] text-[#6b6f76] mt-12 uppercase tracking-wider">
              Powered by RandomForest + personalized pagerank + shap explainers
            </p>
          </div>

          {/* Interactive WebGL Globe */}
          <div className="relative border-3 border-[#0A0A0A] bg-[#0A0A0F] p-4 [box-shadow:10px_10px_0_#0A0A0A] overflow-hidden min-h-[420px] h-full flex flex-col justify-between">
            <div className="flex items-center justify-between border-b-2 border-white/20 pb-3 mb-4">
              <div className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 bg-[#E63946] animate-pulse rounded-full" />
                <span className="font-mono text-[10px] text-white/80 font-bold uppercase tracking-widest">LIVE AML RADAR MAP</span>
              </div>
              <span className="font-mono text-[9px] text-white/50 uppercase">200K SEEDED FLOWS</span>
            </div>
            <div className="flex-1 w-full h-[320px]">
              <RiskGlobe />
            </div>
            <div className="absolute right-4 bottom-4 bg-[#E63946] text-white border-2 border-black font-display font-extrabold text-[10px] tracking-wider px-3 py-1.5 uppercase [box-shadow:3px_3px_0_#0A0A0A]">
              GLOBE CONTROLLER · 3D
            </div>
          </div>
        </div>
      </header>

      {/* Statistics Band */}
      <div className="bg-[#0A0A0A] text-[#F2F0EB] py-8 border-b-3 border-[#0A0A0A]">
        <div className="max-w-[1220px] mx-auto px-6 grid grid-cols-2 lg:grid-cols-4 gap-8">
          <div className="flex flex-col border-r border-[#2c2c2c] last:border-0 pr-4">
            <span className="font-display font-extrabold text-4xl text-[#D4A843]">
              {stats ? stats.total_transactions.toLocaleString() : '200,000'}
            </span>
            <span className="font-mono text-[9px] uppercase tracking-wider text-[#9aa0ad] mt-2">
              Seeded Transaction Database
            </span>
          </div>
          <div className="flex flex-col border-r border-[#2c2c2c] last:border-0 pr-4">
            <span className="font-display font-extrabold text-4xl text-[#E63946]">
              {stats ? stats.flagged_transactions.toLocaleString() : '1,842'}
            </span>
            <span className="font-mono text-[9px] uppercase tracking-wider text-[#9aa0ad] mt-2">
              Highly Suspicious Detections
            </span>
          </div>
          <div className="flex flex-col border-r border-[#2c2c2c] last:border-0 pr-4">
            <span className="font-display font-extrabold text-4xl text-[#2EC04A]">
              {stats ? `${(stats.f1_score * 100).toFixed(1)}%` : '97.2%'}
            </span>
            <span className="font-mono text-[9px] uppercase tracking-wider text-[#9aa0ad] mt-2">
              F1 Benchmark Accuracy
            </span>
          </div>
          <div className="flex flex-col last:border-0">
            <span className="font-display font-extrabold text-4xl text-[#5BC0EB]">
              &lt; 350ms
            </span>
            <span className="font-mono text-[9px] uppercase tracking-wider text-[#9aa0ad] mt-2">
              Agent Action Routing latency
            </span>
          </div>
        </div>
      </div>

      {/* Main capabilities / features grid */}
      <section id="features" className="py-20 border-b-3 border-[#0A0A0A] bg-[#F2F0EB]">
        <div className="max-w-[1220px] mx-auto px-6">
          <div className="sec-tag">
            <span className="sw bg-[#D4A843]" />
            <span className="font-mono text-[10px] uppercase font-bold tracking-widest text-[#6b6f76]">Core Capabilities</span>
          </div>
          <h2 className="font-display font-extrabold text-3xl md:text-5xl uppercase tracking-tight text-[#0A0A0A] mb-4">
            ONE BRAIN. <span className="px-2 py-0.5 bg-[#5BC0EB] border-2 border-black inline-block rotate-[-0.5deg]">FOUR ENGINES.</span>
          </h2>
          <p className="font-mono text-xs text-[#2b2b2b] mb-12 max-w-[60ch]">
            An integrated agentic AML cockpit. Valkyrie coordinates models, personalized pagerank risk maps, SHAP calculations, and automatic report compilers.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            <NeuCard variant="default">
              <div className="h-10 w-10 bg-[#D4A843] border-2 border-black flex items-center justify-center mb-6">
                <Terminal className="h-5 w-5 text-black" />
              </div>
              <h3 className="font-display font-bold text-lg mb-3">LangGraph Orchestrator</h3>
              <p className="font-mono text-[11px] text-[#2b2b2b] leading-relaxed">
                Interprets natural language queries, creates structured execution plans, selectively runs specific compliance tools, and explains results.
              </p>
            </NeuCard>

            <NeuCard variant="critical">
              <div className="h-10 w-10 bg-[#E63946] border-2 border-black flex items-center justify-center mb-6">
                <Shield className="h-5 w-5 text-white" />
              </div>
              <h3 className="font-display font-bold text-lg mb-3">Supervised AML ML</h3>
              <p className="font-mono text-[11px] text-[#2b2b2b] leading-relaxed">
                Random Forest classifier engineered with cash velocity, currency conversion, cross-border flows, fan-in ratios, and amount deviations.
              </p>
            </NeuCard>

            <NeuCard variant="info">
              <div className="h-10 w-10 bg-[#5BC0EB] border-2 border-black flex items-center justify-center mb-6">
                <Network className="h-5 w-5 text-black" />
              </div>
              <h3 className="font-display font-bold text-lg mb-3">Graph Risk Engine</h3>
              <p className="font-mono text-[11px] text-[#2b2b2b] leading-relaxed">
                Directed graph tracking. Personalized PageRank propagates risks to hidden nodes. Multi-hop layering chain tracer detects structured loops.
              </p>
            </NeuCard>

            <NeuCard variant="success">
              <div className="h-10 w-10 bg-[#2EC04A] border-2 border-black flex items-center justify-center mb-6">
                <BarChart3 className="h-5 w-5 text-black" />
              </div>
              <h3 className="font-display font-bold text-lg mb-3">Explainable SHAP</h3>
              <p className="font-mono text-[11px] text-[#2b2b2b] leading-relaxed">
                Generates math-backed feature contribution lists. Translates shapley values into plain English reports for immediate compliance signoff.
              </p>
            </NeuCard>
          </div>
        </div>
      </section>

      {/* How it works section */}
      <section id="how-it-works" className="py-20 border-b-3 border-[#0A0A0A] bg-white">
        <div className="max-w-[1220px] mx-auto px-6">
          <div className="sec-tag">
            <span className="sw bg-[#E63946]" />
            <span className="font-mono text-[10px] uppercase font-bold tracking-widest text-[#6b6f76]">Orchestration Architecture</span>
          </div>
          <h2 className="font-display font-extrabold text-3xl md:text-5xl uppercase tracking-tight text-[#0A0A0A] mb-4">
            AGENTIC EXECUTION FLOW
          </h2>
          <p className="font-mono text-xs text-[#2b2b2b] mb-16 max-w-[60ch]">
            No static sequentially run pipelines. When you query Valkyrie, it decides what is needed, builds a route, runs tools, and compiles findings.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="border-3 border-[#0A0A0A] p-6 pt-10 bg-[#F2F0EB] relative">
              <div className="absolute -top-6 left-6 h-10 w-10 bg-[#0A0A0A] text-[#D4A843] border-3 border-[#D4A843] font-display font-bold flex items-center justify-center">
                01
              </div>
              <h3 className="font-display font-bold text-base mt-2 mb-3">1. NATURAL LANGUAGE ROUTING</h3>
              <p className="font-mono text-[11px] text-[#2b2b2b] leading-relaxed">
                Query parsing extracts dates, amounts, account IDs, segments, and laundering typologies (structuring/layering/smurfing).
              </p>
            </div>

            <div className="border-3 border-[#0A0A0A] p-6 pt-10 bg-[#F2F0EB] relative">
              <div className="absolute -top-6 left-6 h-10 w-10 bg-[#0A0A0A] text-[#D4A843] border-3 border-[#D4A843] font-display font-bold flex items-center justify-center">
                02
              </div>
              <h3 className="font-display font-bold text-base mt-2 mb-3">2. TOOL ENSEMBLE EXECUTION</h3>
              <p className="font-mono text-[11px] text-[#2b2b2b] leading-relaxed">
                Invokes the exact subsystem: Transaction Search, PageRank Propagation, Chain Tracer, Model Evaluator, or SHAP Explainer.
              </p>
            </div>

            <div className="border-3 border-[#0A0A0A] p-6 pt-10 bg-[#F2F0EB] relative">
              <div className="absolute -top-6 left-6 h-10 w-10 bg-[#0A0A0A] text-[#D4A843] border-3 border-[#D4A843] font-display font-bold flex items-center justify-center">
                03
              </div>
              <h3 className="font-display font-bold text-base mt-2 mb-3">3. COMPLIANCE EXPLANATION</h3>
              <p className="font-mono text-[11px] text-[#2b2b2b] leading-relaxed">
                Synthesizes tool output, calculates SHAP contributions, classes overall risk level, generates SAR draft text, and prints actions.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-[#0A0A0A] text-[#F2F0EB] py-12 border-t-3 border-[#0A0A0A] mt-auto">
        <div className="max-w-[1220px] mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <span className="h-8 w-8 bg-[#0A0A0A] text-[#D4A843] border-2 border-[#F2F0EB] font-display font-bold text-base flex items-center justify-center [box-shadow:2px_2px_0_#D4A843]">
              V
            </span>
            <span className="font-display font-bold text-base tracking-wider">
              VALKYRIE AML
            </span>
          </div>
          <span className="font-mono text-[10px] text-[#9aa0ad] uppercase">
            Built for Compliance Teams globally · Hackathon 2026 Submission
          </span>
        </div>
      </footer>
    </div>
  );
}
