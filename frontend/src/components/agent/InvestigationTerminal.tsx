'use client';

import React, { useState, useEffect, useRef } from 'react';
import { api, AnomalyItem } from '@/lib/api';
import { NeuCard } from '../shared/NeuCard';
import { NeuBtn } from '../shared/NeuBtn';
import { RiskBadge } from '../shared/RiskBadge';
import { Terminal, ShieldAlert, CheckCircle2, Play, ChevronRight, HelpCircle } from 'lucide-react';

interface TerminalLine {
  text: string;
  type: 'prompt' | 'info' | 'success' | 'warn' | 'error';
}

interface InvestigationTerminalProps {
  onInvestigationComplete?: (results: any) => void;
  activeQuery?: string;
  triggerSearch?: boolean;
}

export const InvestigationTerminal: React.FC<InvestigationTerminalProps> = ({
  onInvestigationComplete,
  activeQuery = '',
  triggerSearch = false,
}) => {
  const [query, setQuery] = useState('');
  const [lines, setLines] = useState<TerminalLine[]>([
    { text: 'Valkyrie Compliance Intelligence Console v2.0.0 initialized.', type: 'info' },
    { text: 'Awaiting instruction query...', type: 'info' },
  ]);
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [planTools, setPlanTools] = useState<Array<{ name: string; status: 'pending' | 'active' | 'done' }>>([]);
  const terminalEndRef = useRef<HTMLDivElement>(null);

  const sampleQueries = [
    'Find structuring patterns in last 30 days',
    'Which accounts made 10+ transactions under $10,000?',
    'Is customer ID 207936746 suspicious?',
    'Evaluate model performance and precision/recall metrics',
  ];

  useEffect(() => {
    if (triggerSearch && activeQuery) {
      setQuery(activeQuery);
      runInvestigation(activeQuery);
    }
  }, [triggerSearch, activeQuery]);

  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [lines]);

  const addLine = (text: string, type: 'prompt' | 'info' | 'success' | 'warn' | 'error' = 'info') => {
    setLines((prev) => [...prev, { text, type }]);
  };

  const runInvestigation = async (queryText: string) => {
    if (!queryText.trim()) return;
    setLoading(true);
    setCurrentStep(1);
    setPlanTools([]);
    
    setLines([
      { text: `query: "${queryText}"`, type: 'prompt' },
      { text: 'Valkyrie orchestrator streaming started.', type: 'info' },
    ]);

    let abortStream: (() => void) | null = null;

    try {
      abortStream = api.investigateStream(
        queryText,
        0.5,
        (event, data) => {
          if (event === 'start') {
            addLine('Connecting to LangGraph agent executor...', 'info');
          } else if (event === 'intent') {
            addLine('Step 1: Extracting query intent & segment filters...', 'info');
            setCurrentStep(1);
          } else if (event === 'intent_detected') {
            addLine(`>> Target typology: ${data.pattern}`, 'success');
            if (Object.keys(data.filters).length > 0) {
              addLine(`>> Filters matched: ${JSON.stringify(data.filters)}`, 'success');
            }
            if (data.tools) {
              setPlanTools(data.tools.map((t: string) => ({ name: t, status: 'pending' })));
            }
          } else if (event === 'planning') {
            addLine('Step 2: Grok-3 drafting structured tool sequence...', 'info');
            setCurrentStep(2);
          } else if (event === 'plan_ready') {
            addLine(`>> Execution sequence formulated:`, 'success');
            data.plan.tools.forEach((t: any, i: number) => {
              addLine(`   [${i + 1}] ${t.name}(${JSON.stringify(t.params)})`, 'success');
            });
            setPlanTools(data.plan.tools.map((t: any) => ({ name: t.name, status: 'pending' })));
          } else if (event === 'tool_start') {
            setCurrentStep(3);
            setPlanTools((prev) =>
              prev.map((t) => (t.name === data.tool ? { ...t, status: 'active' } : t))
            );
            addLine(`[Tool Dispatch] Executing: ${data.tool}...`, 'warn');
          } else if (event === 'tool_done') {
            setPlanTools((prev) =>
              prev.map((t) => (t.name === data.tool ? { ...t, status: 'done' } : t))
            );
            addLine(`[Tool Finished] ${data.tool} execution complete.`, 'success');
          } else if (event === 'classifying') {
            addLine('Step 4: Classifying risk levels & running SHAP explainers...', 'info');
            setCurrentStep(4);
          } else if (event === 'classified') {
            addLine('>> Account risk thresholds evaluated successfully.', 'success');
          } else if (event === 'summarizing') {
            addLine('Step 5: Invoking Grok-3 to compile compliance report summary...', 'info');
            setCurrentStep(5);
          } else if (event === 'complete') {
            addLine('>> Compliance investigation complete.', 'success');
            addLine(`>> Summary: ${data.summary}`, 'success');
            setLoading(false);
            setCurrentStep(5);
            if (onInvestigationComplete) {
              onInvestigationComplete(data);
            }
          }
        },
        (err) => {
          console.error(err);
          addLine('API communication error. Using offline fallback metrics.', 'error');
          setLoading(false);
        }
      );
    } catch (e: any) {
      addLine(`Orchestrator failure: ${e.message}`, 'error');
      setLoading(false);
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    runInvestigation(query);
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Dynamic agent plan flow indicators */}
      <div className="grid grid-cols-5 gap-4">
        {[
          { step: 1, label: 'INTENT' },
          { step: 2, label: 'PLAN' },
          { step: 3, label: 'EXECUTE' },
          { step: 4, label: 'RISK' },
          { step: 5, label: 'SUMMARY' },
        ].map((s) => (
          <div
            key={s.step}
            className={`border-3 border-black p-3 text-center transition-all duration-150 ${
              currentStep === s.step
                ? 'bg-[#D4A843] [box-shadow:3px_3px_0_#0A0A0A] -translate-x-[1.5px] -translate-y-[1.5px]'
                : currentStep > s.step
                ? 'bg-[#2EC04A] [box-shadow:3px_3px_0_#0A0A0A]'
                : 'bg-white text-black/40 opacity-70'
            }`}
          >
            <div className="font-display font-extrabold text-xs">{s.label}</div>
            <div className="font-mono text-[9px] mt-1 font-bold">STEP 0{s.step}</div>
          </div>
        ))}
      </div>

      {/* Terminal Display */}
      <div className="terminal rounded-none min-h-[340px] flex flex-col justify-between border-3 border-black [box-shadow:6px_6px_0_#0A0A0A]">
        <div className="flex-1 overflow-y-auto max-h-[380px] pr-2">
          {lines.map((l, i) => (
            <div
              key={i}
              className={`terminal-line ${
                l.type === 'prompt'
                  ? 'terminal-prompt text-[#D4A843] font-bold'
                  : l.type === 'success'
                  ? 'text-[#2EC04A]'
                  : l.type === 'error'
                  ? 'text-[#E63946]'
                  : l.type === 'warn'
                  ? 'text-[#F97316]'
                  : 'text-[#F2F0EB]/95'
              }`}
            >
              {l.text}
            </div>
          ))}
          {loading && <div className="terminal-line cursor text-[#D4A843] font-bold">Analyzing pipeline</div>}
          <div ref={terminalEndRef} />
        </div>

        {/* Current Tool execution tracker */}
        {planTools.length > 0 && (
          <div className="border-t border-white/20 pt-3 mt-3 flex flex-wrap gap-3">
            <span className="font-display text-[9px] uppercase tracking-widest text-[#9aa0ad] mr-1 flex items-center">
              Active Plan:
            </span>
            {planTools.map((t, idx) => (
              <div
                key={idx}
                className={`tool-node border-2 border-white/20 ${
                  t.status === 'active'
                    ? 'bg-[#D4A843] text-black border-black font-bold animate-pulse'
                    : t.status === 'done'
                    ? 'bg-[#2EC04A] text-black border-black font-bold'
                    : 'bg-transparent text-white/50'
                }`}
              >
                {t.name}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Query Input form */}
      <form onSubmit={handleFormSubmit} className="flex gap-4">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Investigate structuring patterns..."
          className="flex-1 border-3 border-black bg-white px-4 py-3 font-mono text-sm focus:outline-none focus:ring-0 focus:border-black placeholder-black/40 [box-shadow:3px_3px_0_#0A0A0A]"
          disabled={loading}
        />
        <NeuBtn variant="ink" type="submit" disabled={loading}>
          <Play className="h-4 w-4" />
          Investigate
        </NeuBtn>
      </form>

      {/* Suggested / Sample Queries */}
      <div className="flex flex-col gap-2">
        <span className="font-display text-[9px] font-bold uppercase tracking-wider text-[#6b6f76]">
          Suggested Investigations:
        </span>
        <div className="flex flex-wrap gap-2">
          {sampleQueries.map((q, i) => (
            <button
              key={i}
              type="button"
              onClick={() => {
                setQuery(q);
                runInvestigation(q);
              }}
              className="text-left font-mono text-[10px] bg-white border-2 border-black px-3 py-1.5 [box-shadow:2.5px_2.5px_0_#0A0A0A] hover:-translate-x-[1.5px] hover:-translate-y-[1.5px] hover:[box-shadow:3.5px_3.5px_0_#0A0A0A] active:translate-x-[1.5px] active:translate-y-[1.5px]"
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
