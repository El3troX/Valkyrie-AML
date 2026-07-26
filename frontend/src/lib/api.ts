const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface DashboardStats {
  total_transactions: number;
  flagged_transactions: number;
  f1_score: number;
  precision: number;
  recall: number;
  false_positives: number;
  optimal_threshold: number;
  laundering_typologies: Record<string, number>;
  avg_anomaly_score: number;
  top_risk_countries: Array<{ country: string; max_risk: number; avg_risk: number; count: number }>;
  suspicious_count: number;
}

export interface AnomalyItem {
  idx: number;
  score: number;
  risk_level: string;
  escalation: string;
  sender: string;
  receiver: string;
  amount: number;
  payment_type: string;
  laundering_type: string;
  date: string;
  sender_location: string;
  receiver_location: string;
  payment_currency: string;
  received_currency: string;
  is_cross_currency: boolean;
}

export interface NetworkNode {
  id: string;
  risk_score: number;
  pagerank: number;
  color: string;
  size: number;
  label: string;
}

export interface NetworkEdge {
  source: string;
  target: string;
  amount: number;
  count: number;
}

export interface NetworkData {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
}

export interface GeoArc {
  from_country: string;
  to_country: string;
  from_lat: number;
  from_lng: number;
  to_lat: number;
  to_lng: number;
  total_amount: number;
  count: number;
  max_risk: number;
  color: string;
}

export interface ModelPerformance {
  threshold: number;
  optimal_threshold: number;
  precision: number;
  recall: number;
  f1: number;
  false_positives: number;
  false_negatives: number;
  true_positives: number;
  true_negatives: number;
  n_actual_suspicious: number;
  n_predicted_suspicious: number;
  confusion_matrix: number[][];
  by_typology: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  illustrative_cases: Array<{
    Sender_account?: string;
    Receiver_account?: string;
    Amount?: number;
    Laundering_type?: string;
    anomaly_score?: number;
    [key: string]: unknown;
  }>;
}

export interface SARDocument {
  account_id: string;
  narrative: string;
  anomaly_data: {
    account_id: string;
    n_transactions: number;
    total_amount: number;
    mean_score: number;
    max_score: number;
  };
  risk_level: string;
  escalation: string;
  shap_explanation: Record<string, unknown>;
  top_connections: Array<{ account: string; ppr_score: number }>;
  transactions: Record<string, unknown>[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

export const api = {
  investigateStream: (
    query: string,
    threshold: number,
    onEvent: (event: string, data: any) => void,
    onError: (err: Error) => void,
  ): (() => void) => {
    let aborted = false;
    const controller = new AbortController();

    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/investigate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, threshold }),
          signal: controller.signal,
        });
        if (!res.ok || !res.body) {
          throw new Error(`API ${res.status}`);
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let currentEvent = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() ?? '';
          for (const line of lines) {
            if (line.startsWith('event:')) {
              currentEvent = line.slice(6).trim();
            } else if (line.startsWith('data:') && currentEvent) {
              try {
                const data = JSON.parse(line.slice(5).trim());
                onEvent(currentEvent, data);
              } catch {}
              currentEvent = '';
            }
          }
        }
      } catch (err: any) {
        if (err.name !== 'AbortError') onError(err);
      }
    })();

    return () => {
      aborted = true;
      controller.abort();
    };
  },

  getDashboardStats: () => request<DashboardStats>('/api/dashboard-stats'),

  getTopAnomalies: (n = 20, threshold = 0) =>
    request<{ anomalies: AnomalyItem[]; total: number }>(
      `/api/top-anomalies?n=${n}&threshold=${threshold}`
    ),

  getNetworkData: (maxNodes = 60) =>
    request<NetworkData>(`/api/network-data?max_nodes=${maxNodes}`),

  generateSAR: (accountId: string) =>
    request<SARDocument>('/api/generate-sar', {
      method: 'POST',
      body: JSON.stringify({ account_id: accountId }),
    }),

  getModelPerformance: (threshold: number) =>
    request<ModelPerformance>(`/api/model-performance?threshold=${threshold}`),

  getGeoData: () => request<{ arcs: GeoArc[] }>('/api/geo-data'),
};
