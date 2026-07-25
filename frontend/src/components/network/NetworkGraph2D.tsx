'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { NetworkData, NetworkNode, NetworkEdge } from '@/lib/api';

interface NetworkGraph2DProps {
  data: NetworkData;
  onNodeClick?: (accountId: string) => void;
}

interface SimNode extends NetworkNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
  fx?: number;
  fy?: number;
}

interface SimEdge extends NetworkEdge {
  sourceNode: SimNode;
  targetNode: SimNode;
}

const RISK_COLORS: Record<string, string> = {
  '#E63946': '#E63946',
  '#F97316': '#F97316',
  '#EAB308': '#EAB308',
  '#2EC04A': '#2EC04A',
};

function getRiskLabel(color: string): string {
  if (color === '#E63946') return 'CRITICAL ≥0.8';
  if (color === '#F97316') return 'HIGH ≥0.6';
  if (color === '#EAB308') return 'MEDIUM ≥0.4';
  return 'LOW <0.4';
}

function formatAmount(amount: number): string {
  if (amount >= 1_000_000) return `$${(amount / 1_000_000).toFixed(1)}M`;
  if (amount >= 1_000) return `$${(amount / 1_000).toFixed(0)}K`;
  return `$${amount.toFixed(0)}`;
}

export const NetworkGraph2D: React.FC<NetworkGraph2DProps> = ({ data, onNodeClick }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animFrameRef = useRef<number>(0);
  const nodesRef = useRef<SimNode[]>([]);
  const edgesRef = useRef<SimEdge[]>([]);
  const transformRef = useRef({ x: 0, y: 0, scale: 1 });
  const dragRef = useRef<{ dragging: boolean; startX: number; startY: number; dragNode: SimNode | null }>({
    dragging: false, startX: 0, startY: 0, dragNode: null,
  });
  const hoveredNodeRef = useRef<SimNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<SimNode | null>(null);
  const [showLegend, setShowLegend] = useState(true);
  const tickRef = useRef(0);

  // Initialize force simulation
  const initSim = useCallback(() => {
    if (!containerRef.current || !data.nodes.length) return;

    const W = containerRef.current.clientWidth || 900;
    const H = containerRef.current.clientHeight || 600;

    // Position nodes in a sunflower pattern for better initial spread
    const nodes: SimNode[] = data.nodes.map((n, i) => {
      const goldenAngle = Math.PI * (3 - Math.sqrt(5));
      const angle = i * goldenAngle;
      const r = Math.min(W, H) * 0.32 * Math.sqrt(i / Math.max(data.nodes.length, 1));
      return {
        ...n,
        x: W / 2 + r * Math.cos(angle) + (Math.random() - 0.5) * 20,
        y: H / 2 + r * Math.sin(angle) + (Math.random() - 0.5) * 20,
        vx: 0,
        vy: 0,
      };
    });

    const nodeMap = new Map(nodes.map(n => [n.id, n]));

    const edges: SimEdge[] = data.edges
      .map(e => {
        const s = nodeMap.get(e.source);
        const t = nodeMap.get(e.target);
        if (!s || !t) return null;
        return { ...e, sourceNode: s, targetNode: t };
      })
      .filter(Boolean) as SimEdge[];

    nodesRef.current = nodes;
    edgesRef.current = edges;
    tickRef.current = 0;

    // Center transform
    transformRef.current = { x: 0, y: 0, scale: 1 };
  }, [data]);

  // Force simulation tick
  const tick = useCallback(() => {
    const nodes = nodesRef.current;
    const edges = edgesRef.current;
    if (!nodes.length) return;

    // Cool down faster for stable layout
    const alpha = Math.max(0.001, 0.35 * Math.exp(-tickRef.current * 0.012));
    tickRef.current++;

    // Repulsion between all nodes — stronger to prevent overlap
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i];
        const b = nodes[j];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 0.1;
        const minDist = getNodeRadius(a) + getNodeRadius(b) + 40;
        // Hard collision + repulsion
        const repulsion = (dist < minDist ? 8000 : 5000) / (dist * dist);
        const fx = (dx / dist) * repulsion;
        const fy = (dy / dist) * repulsion;
        a.vx -= fx * alpha;
        a.vy -= fy * alpha;
        b.vx += fx * alpha;
        b.vy += fy * alpha;
      }
    }

    // Spring attraction along edges — longer rest length
    for (const edge of edges) {
      const s = edge.sourceNode;
      const t = edge.targetNode;
      const dx = t.x - s.x;
      const dy = t.y - s.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const targetDist = 160; // wider spacing
      const spring = (dist - targetDist) * 0.03 * alpha;
      const fx = (dx / dist) * spring;
      const fy = (dy / dist) * spring;
      s.vx += fx;
      s.vy += fy;
      t.vx -= fx;
      t.vy -= fy;
    }

    // Centering / gravity force
    const W = canvasRef.current?.width || 900;
    const H = canvasRef.current?.height || 600;
    for (const n of nodes) {
      n.vx += (W / 2 - n.x) * 0.004 * alpha;
      n.vy += (H / 2 - n.y) * 0.004 * alpha;
    }

    // Integrate positions with stronger damping for stability
    for (const n of nodes) {
      if (n.fx !== undefined) { n.x = n.fx; n.y = n.fy!; continue; }
      n.vx *= 0.65;
      n.vy *= 0.65;
      n.x += n.vx;
      n.y += n.vy;
    }
  }, []);

  // Draw frame
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const nodes = nodesRef.current;
    const edges = edgesRef.current;
    const { x: tx, y: ty, scale } = transformRef.current;

    const W = canvas.width;
    const H = canvas.height;

    // Background
    ctx.fillStyle = '#0A0A0F';
    ctx.fillRect(0, 0, W, H);

    // Grid dots
    ctx.fillStyle = 'rgba(212,168,67,0.06)';
    const gridSize = 40 * scale;
    const offsetX = (tx % gridSize + gridSize) % gridSize;
    const offsetY = (ty % gridSize + gridSize) % gridSize;
    for (let gx = offsetX; gx < W; gx += gridSize) {
      for (let gy = offsetY; gy < H; gy += gridSize) {
        ctx.beginPath();
        ctx.arc(gx, gy, 1, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    ctx.save();
    ctx.translate(tx, ty);
    ctx.scale(scale, scale);

    // Draw edges
    for (const edge of edges) {
      const s = edge.sourceNode;
      const t = edge.targetNode;
      
      // Node radii
      const rS = getNodeRadius(s);
      const rT = getNodeRadius(t);

      // Determine if there is a bidirectional link between these two nodes
      const hasBi = edges.some(e => e.source === edge.target && e.target === edge.source);
      // Consistent order to determine curve direction
      const isReverse = edge.source > edge.target;

      let x1 = s.x;
      let y1 = s.y;
      let x2 = t.x;
      let y2 = t.y;

      let mx = (x1 + x2) / 2;
      let my = (y1 + y2) / 2;
      let cpX = mx;
      let cpY = my;

      if (hasBi) {
        // Curve offset normal to the line
        const dxLine = x2 - x1;
        const dyLine = y2 - y1;
        const distLine = Math.sqrt(dxLine * dxLine + dyLine * dyLine) || 0.1;
        const nxNormal = -dyLine / distLine;
        const nyNormal = dxLine / distLine;
        
        // Offset cpX and cpY to create a quadratic curve
        const curveOffset = 30; // curve offset height
        const dir = isReverse ? 1 : -1;
        cpX = mx + nxNormal * curveOffset * dir;
        cpY = my + nyNormal * curveOffset * dir;

        // Recalculate true midpoint of the quadratic bezier curve for the label
        mx = 0.25 * x1 + 0.5 * cpX + 0.25 * x2;
        my = 0.25 * y1 + 0.5 * cpY + 0.25 * y2;
      }

      // Calculate tangent vectors at start and end for node offsets and arrows
      const dxStart = cpX - x1;
      const dyStart = cpY - y1;
      const distStart = Math.sqrt(dxStart * dxStart + dyStart * dyStart) || 0.1;
      
      const dxEnd = x2 - cpX;
      const dyEnd = y2 - cpY;
      const distEnd = Math.sqrt(dxEnd * dxEnd + dyEnd * dyEnd) || 0.1;

      // Adjust endpoints to sit cleanly on node borders
      const px1 = x1 + (dxStart / distStart) * rS;
      const py1 = y1 + (dyStart / distStart) * rS;
      const px2 = x2 - (dxEnd / distEnd) * (rT + 6);
      const py2 = y2 - (dyEnd / distEnd) * (rT + 6);

      // Edge styling
      const isHighRisk = s.color === '#E63946' || t.color === '#E63946';
      const edgeColor = isHighRisk ? '#E63946' : 'rgba(100,120,160,0.35)';

      ctx.beginPath();
      ctx.moveTo(px1, py1);
      if (hasBi) {
        ctx.quadraticCurveTo(cpX, cpY, px2, py2);
      } else {
        ctx.lineTo(px2, py2);
      }
      ctx.strokeStyle = edgeColor;
      ctx.lineWidth = isHighRisk ? 1.5 / scale : 1.0 / scale;
      
      // Animated dashed effect for high risk edges
      if (isHighRisk) {
        ctx.setLineDash([4 / scale, 4 / scale]);
        ctx.lineDashOffset = -tickRef.current * 0.2;
      } else {
        ctx.setLineDash([]);
      }
      ctx.stroke();
      ctx.setLineDash([]); // Reset line dash

      // Draw arrowhead at px2, py2 aligned with tangent
      const arrowLen = 8 / scale;
      const arrowAngle = 0.4;
      const nxEnd = dxEnd / distEnd;
      const nyEnd = dyEnd / distEnd;
      const ax = px2 - nxEnd * arrowLen;
      const ay = py2 - nyEnd * arrowLen;
      const angle = Math.atan2(nyEnd, nxEnd);

      ctx.beginPath();
      ctx.moveTo(px2, py2);
      ctx.lineTo(
        px2 - Math.cos(angle - arrowAngle) * arrowLen,
        py2 - Math.sin(angle - arrowAngle) * arrowLen
      );
      ctx.lineTo(
        px2 - Math.cos(angle + arrowAngle) * arrowLen,
        py2 - Math.sin(angle + arrowAngle) * arrowLen
      );
      ctx.closePath();
      ctx.fillStyle = edgeColor;
      ctx.fill();

      // Edge label (amount) — only show if scale is reasonable
      if (scale > 0.45) {
        const label = formatAmount(edge.amount);
        const fontSize = Math.max(8, Math.min(10, 8 / scale));
        ctx.font = `bold ${fontSize}px "JetBrains Mono", monospace`;
        const tw = ctx.measureText(label).width;

        // Neubrutalist small label pill
        ctx.fillStyle = '#0A0A0F';
        ctx.fillRect(mx - tw / 2 - 3, my - fontSize / 2 - 2, tw + 6, fontSize + 4);

        ctx.strokeStyle = isHighRisk ? '#E63946' : 'rgba(212,168,67,0.45)';
        ctx.lineWidth = 1 / scale;
        ctx.strokeRect(mx - tw / 2 - 3, my - fontSize / 2 - 2, tw + 6, fontSize + 4);

        ctx.fillStyle = isHighRisk ? '#E63946' : '#D4A843';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, mx, my);
      }
    }

    // Draw nodes
    for (const node of nodes) {
      const r = getNodeRadius(node);
      const isHovered = hoveredNodeRef.current?.id === node.id;

      // Glow effect for high-risk nodes (layering hubs)
      if (node.color === '#E63946' || node.color === '#F97316') {
        const glow = ctx.createRadialGradient(node.x, node.y, r * 0.4, node.x, node.y, r * 2.2);
        glow.addColorStop(0, node.color + '33');
        glow.addColorStop(1, 'transparent');
        ctx.beginPath();
        ctx.arc(node.x, node.y, r * 2.2, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();
      }

      // Neubrutalist drop shadow (solid offset black circle)
      ctx.beginPath();
      ctx.arc(node.x + 3.5 / scale, node.y + 3.5 / scale, r, 0, Math.PI * 2);
      ctx.fillStyle = '#0A0A0A';
      ctx.fill();

      // Node body
      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
      ctx.fillStyle = node.color;
      ctx.fill();

      // Hard black border (neubrutualism style)
      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
      ctx.strokeStyle = isHovered ? '#D4A843' : '#0A0A0A';
      ctx.lineWidth = isHovered ? 3 / scale : 2.2 / scale;
      ctx.stroke();

      // Hover dash ring
      if (isHovered) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, r + 6 / scale, 0, Math.PI * 2);
        ctx.strokeStyle = '#D4A843';
        ctx.lineWidth = 2 / scale;
        ctx.setLineDash([4 / scale, 3 / scale]);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Node label (truncated account ID)
      if (scale > 0.3) {
        const fontSize = Math.max(7, Math.min(10, 8.5 / scale));
        ctx.font = `bold ${fontSize}px "JetBrains Mono", monospace`;
        const shortId = node.id.length > 10 ? node.id.slice(0, 10) : node.id;
        const labelW = ctx.measureText(shortId).width;

        // Label pill background
        const ly = node.y + r + 11 / scale;
        ctx.fillStyle = 'rgba(10,10,15,0.95)';
        ctx.fillRect(node.x - labelW / 2 - 4, ly - fontSize / 2 - 2, labelW + 8, fontSize + 4);
        
        ctx.strokeStyle = node.color;
        ctx.lineWidth = 1 / scale;
        ctx.strokeRect(node.x - labelW / 2 - 4, ly - fontSize / 2 - 2, labelW + 8, fontSize + 4);

        ctx.fillStyle = '#F2F0EB';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(shortId, node.x, ly);
      }
    }

    ctx.restore();

    // --- Hover tooltip (canvas-space overlay) ---
    const hn = hoveredNodeRef.current;
    if (hn) {
      const sx = hn.x * scale + tx;
      const sy = hn.y * scale + ty;

      const lines = [
        `ACC: ${hn.id}`,
        `RISK: ${(hn.risk_score * 100).toFixed(1)}%`,
        `PPR: ${(hn.pagerank * 100).toFixed(2)}%`,
      ];
      const tooltipW = 200;
      const tooltipH = lines.length * 18 + 20;
      let ttx = sx + 20;
      let tty = sy - tooltipH / 2;
      if (ttx + tooltipW > W) ttx = sx - tooltipW - 20;
      if (tty < 4) tty = 4;
      if (tty + tooltipH > H) tty = H - tooltipH - 4;

      // Neubrutualism tooltip
      ctx.fillStyle = '#F2F0EB';
      ctx.fillRect(ttx, tty, tooltipW, tooltipH);
      ctx.strokeStyle = '#0A0A0A';
      ctx.lineWidth = 2.5;
      ctx.strokeRect(ttx, tty, tooltipW, tooltipH);
      // Offset shadow
      ctx.fillStyle = '#0A0A0A';
      ctx.fillRect(ttx + 4, tty + tooltipH, tooltipW, 4);
      ctx.fillRect(ttx + tooltipW, tty + 4, 4, tooltipH);

      // Colored top bar
      ctx.fillStyle = hn.color;
      ctx.fillRect(ttx, tty, tooltipW, 22);
      ctx.fillStyle = '#0A0A0A';
      ctx.font = 'bold 10px "Space Grotesk", sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(getRiskLabel(hn.color), ttx + 8, tty + 11);

      ctx.fillStyle = '#0A0A0A';
      ctx.font = 'bold 10px "JetBrains Mono", monospace';
      lines.forEach((line, i) => {
        ctx.fillText(line, ttx + 10, tty + 22 + i * 18 + 9);
      });
    }
  }, []);

  function getNodeRadius(node: SimNode): number {
    return Math.max(6, Math.min(28, 6 + node.pagerank * 22));
  }

  // Get node at canvas point (accounting for transform)
  const getNodeAtPoint = useCallback((cx: number, cy: number): SimNode | null => {
    const { x: tx, y: ty, scale } = transformRef.current;
    const wx = (cx - tx) / scale;
    const wy = (cy - ty) / scale;
    const nodes = nodesRef.current;
    for (let i = nodes.length - 1; i >= 0; i--) {
      const n = nodes[i];
      const r = getNodeRadius(n);
      const dx = wx - n.x;
      const dy = wy - n.y;
      if (dx * dx + dy * dy <= (r + 4) * (r + 4)) return n;
    }
    return null;
  }, []);

  // Animation loop
  useEffect(() => {
    if (!data.nodes.length) return;
    initSim();

    const loop = () => {
      tick();
      draw();
      animFrameRef.current = requestAnimationFrame(loop);
    };
    animFrameRef.current = requestAnimationFrame(loop);

    return () => cancelAnimationFrame(animFrameRef.current);
  }, [data, initSim, tick, draw]);

  // Resize observer
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const resize = () => {
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  // Mouse events
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const getPos = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    };

    const onMouseDown = (e: MouseEvent) => {
      const pos = getPos(e);
      const node = getNodeAtPoint(pos.x, pos.y);
      if (node) {
        dragRef.current = { dragging: true, startX: pos.x, startY: pos.y, dragNode: node };
        node.fx = node.x;
        node.fy = node.y;
      } else {
        dragRef.current = { dragging: true, startX: pos.x, startY: pos.y, dragNode: null };
      }
    };

    const onMouseMove = (e: MouseEvent) => {
      const pos = getPos(e);
      const { dragging, startX, startY, dragNode } = dragRef.current;

      if (dragging) {
        if (dragNode) {
          const { scale } = transformRef.current;
          dragNode.fx = dragNode.x + (pos.x - startX) / scale;
          dragNode.fy = dragNode.y + (pos.y - startY) / scale;
          dragNode.x = dragNode.fx;
          dragNode.y = dragNode.fy;
        } else {
          transformRef.current.x += pos.x - startX;
          transformRef.current.y += pos.y - startY;
        }
        dragRef.current.startX = pos.x;
        dragRef.current.startY = pos.y;
      }

      const node = getNodeAtPoint(pos.x, pos.y);
      if (node?.id !== hoveredNodeRef.current?.id) {
        hoveredNodeRef.current = node;
        setHoveredNode(node);
        canvas.style.cursor = node ? 'pointer' : 'grab';
      }
    };

    const onMouseUp = (e: MouseEvent) => {
      const pos = getPos(e);
      const { dragging, startX, startY, dragNode } = dragRef.current;

      // Release pinned node
      if (dragNode) {
        dragNode.fx = undefined;
        dragNode.fy = undefined;
      }

      // Click detection (small movement)
      const dx = pos.x - startX;
      const dy = pos.y - startY;
      if (Math.sqrt(dx * dx + dy * dy) < 5 && dragNode && onNodeClick) {
        onNodeClick(dragNode.id);
      }

      dragRef.current = { dragging: false, startX: 0, startY: 0, dragNode: null };
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const pos = getPos(e);
      const delta = e.deltaY < 0 ? 1.12 : 0.89;
      const newScale = Math.max(0.15, Math.min(4, transformRef.current.scale * delta));
      // Zoom toward mouse position
      transformRef.current.x = pos.x - (pos.x - transformRef.current.x) * (newScale / transformRef.current.scale);
      transformRef.current.y = pos.y - (pos.y - transformRef.current.y) * (newScale / transformRef.current.scale);
      transformRef.current.scale = newScale;
    };

    const onDblClick = (e: MouseEvent) => {
      const pos = getPos(e);
      const node = getNodeAtPoint(pos.x, pos.y);
      if (node && onNodeClick) onNodeClick(node.id);
    };

    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('dblclick', onDblClick);

    return () => {
      canvas.removeEventListener('mousedown', onMouseDown);
      canvas.removeEventListener('mousemove', onMouseMove);
      canvas.removeEventListener('mouseup', onMouseUp);
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('dblclick', onDblClick);
    };
  }, [getNodeAtPoint, onNodeClick]);

  const handleZoomIn = () => {
    transformRef.current.scale = Math.min(4, transformRef.current.scale * 1.25);
  };
  const handleZoomOut = () => {
    transformRef.current.scale = Math.max(0.15, transformRef.current.scale * 0.8);
  };
  const handleReset = () => {
    transformRef.current = { x: 0, y: 0, scale: 1 };
    initSim();
  };

  return (
    <div ref={containerRef} className="relative w-full h-full min-h-[600px] bg-[#0A0A0F] border-3 border-black overflow-hidden">
      <canvas ref={canvasRef} className="w-full h-full cursor-grab" />

      {/* Legend toggle button */}
      <button
        onClick={() => setShowLegend(v => !v)}
        className="absolute top-3 left-3 bg-[#F2F0EB] border-3 border-black [box-shadow:3px_3px_0_#D4A843] font-display font-bold text-[10px] uppercase tracking-wider px-3 py-2 hover:-translate-x-[1px] hover:-translate-y-[1px] transition-transform"
      >
        {showLegend ? '▲ Legend' : '▼ Legend'}
      </button>

      {/* Legend panel */}
      {showLegend && (
        <div className="absolute top-12 left-3 bg-[#F2F0EB] border-3 border-black [box-shadow:4px_4px_0_#0A0A0A] p-4 w-[260px] z-10">
          <div className="border-b-2 border-black pb-2 mb-3">
            <span className="font-display font-extrabold text-xs uppercase tracking-wider">How to Read This Graph</span>
          </div>

          <div className="mb-3">
            <div className="font-display font-bold text-[9px] uppercase tracking-wider text-[#6b6f76] mb-1.5">Node Colors (Risk)</div>
            {[
              { color: '#E63946', label: 'CRITICAL ≥0.8 — Anomaly flag' },
              { color: '#F97316', label: 'HIGH ≥0.6 — Velocity z-score' },
              { color: '#EAB308', label: 'MEDIUM ≥0.4 — Network risk' },
              { color: '#2EC04A', label: 'LOW <0.4 — Baseline normal' },
            ].map(r => (
              <div key={r.color} className="flex items-center gap-2 mb-1">
                <span className="h-3 w-3 rounded-full border-2 border-black flex-shrink-0" style={{ background: r.color }} />
                <span className="font-mono text-[9px]">{r.label}</span>
              </div>
            ))}
          </div>

          <div className="mb-3">
            <div className="font-display font-bold text-[9px] uppercase tracking-wider text-[#6b6f76] mb-1.5">Edges & Sizes</div>
            <div className="font-mono text-[9px] space-y-0.5">
              <div>• <b>Node Size</b>: Personalised PageRank centrality</div>
              <div>• <b>Arrow</b>: Money transfer direction A→B</div>
              <div>• <b>Edge Label</b>: Aggregated $ amount</div>
            </div>
          </div>

          <div>
            <div className="font-display font-bold text-[9px] uppercase tracking-wider text-[#6b6f76] mb-1.5">Visual Patterns</div>
            <div className="font-mono text-[9px] space-y-0.5">
              <div>• <b>Smurfing</b>: Many senders → 1 hub</div>
              <div>• <b>Layering</b>: Chain A→B→C→D</div>
              <div>• <b>Structuring</b>: Many sub-$10K edges</div>
            </div>
          </div>
        </div>
      )}

      {/* Zoom controls (bottom-left) */}
      <div className="absolute bottom-4 left-4 flex flex-col gap-2">
        <button onClick={handleZoomIn} className="h-9 w-9 bg-[#2EC04A] border-3 border-black [box-shadow:2px_2px_0_#0A0A0A] font-bold text-base hover:-translate-x-[1px] hover:-translate-y-[1px] transition-transform">+</button>
        <button onClick={handleZoomOut} className="h-9 w-9 bg-[#E63946] text-white border-3 border-black [box-shadow:2px_2px_0_#0A0A0A] font-bold text-base hover:-translate-x-[1px] hover:-translate-y-[1px] transition-transform">−</button>
        <button onClick={handleReset} className="h-9 w-9 bg-[#D4A843] border-3 border-black [box-shadow:2px_2px_0_#0A0A0A] font-bold text-xs hover:-translate-x-[1px] hover:-translate-y-[1px] transition-transform">↺</button>
      </div>

      {/* Stats overlay (bottom-right) */}
      <div className="absolute bottom-4 right-4 bg-[#0A0A0F] border-2 border-[#D4A843]/60 px-3 py-2">
        <div className="font-mono text-[9px] text-[#D4A843] font-bold">
          {data.nodes.length} NODES · {data.edges.length} EDGES
        </div>
        <div className="font-mono text-[8px] text-white/40 mt-0.5">
          Scroll to zoom · Drag to pan · Click node
        </div>
      </div>
    </div>
  );
};
