"""
Network risk propagation engine for Valkyrie-AML.

Builds a directed transaction graph from SAML-D data and provides
Personalised PageRank risk scoring and multi-hop chain tracing for
layering detection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


class TransactionGraph:
    """Directed graph of the SAML-D transaction network.

    Nodes represent accounts (both sender and receiver).  Edges are
    aggregated transactions with ``weight`` (total amount) and
    ``tx_count`` metadata.

    Parameters
    ----------
    df : pd.DataFrame
        SAML-D transactions.  Must include ``Sender_account``,
        ``Receiver_account``, ``Amount``, ``Date``, ``Time``.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.graph: nx.DiGraph = nx.DiGraph()
        self._time_ordered: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> TransactionGraph:
        """Construct the directed graph from the transaction DataFrame.

        Uses nx.from_pandas_edgelist for speed on large datasets.
        Returns self.
        """
        # Aggregate edges efficiently
        edge_data = (
            self.df.groupby(["Sender_account", "Receiver_account"])
            .agg(weight=("Amount", "sum"), tx_count=("Amount", "count"))
            .reset_index()
        )
        edge_data["Sender_account"] = edge_data["Sender_account"].astype(str)
        edge_data["Receiver_account"] = edge_data["Receiver_account"].astype(str)

        # Build graph with from_pandas_edgelist (much faster than iterrows)
        self.graph = nx.from_pandas_edgelist(
            edge_data, "Sender_account", "Receiver_account",
            edge_attr=["weight", "tx_count"], create_using=nx.DiGraph()
        )

        # Build time-ordered index for chain tracing with fast explicit format parsing
        self.df["dt_ts"] = pd.to_datetime(
            self.df["Date"] + " " + self.df["Time"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        self._time_ordered = self.df.sort_values("dt_ts").reset_index(drop=True)
        self._edges_by_sender: dict[str, list[dict]] | None = None

        return self

    def _build_edge_index(self, min_amount: float = 0.0) -> dict[str, list[dict]]:
        """Fast edge adjacency dict construction using itertuples."""
        if self._edges_by_sender is not None:
            return self._edges_by_sender

        if self._time_ordered is None:
            return {}

        edges_by_sender: dict[str, list[dict]] = {}
        for row in self._time_ordered.itertuples(index=False):
            sender = str(getattr(row, "Sender_account"))
            receiver = str(getattr(row, "Receiver_account"))
            amount = float(getattr(row, "Amount"))
            if amount < min_amount:
                continue
            edges_by_sender.setdefault(sender, []).append({
                "receiver": receiver,
                "amount": amount,
                "timestamp": getattr(row, "dt_ts"),
                "sender": sender,
            })
        self._edges_by_sender = edges_by_sender
        return edges_by_sender

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """Return graph statistics."""
        if self.graph.number_of_nodes() == 0:
            return {
                "n_nodes": 0,
                "n_edges": 0,
                "density": 0.0,
                "n_components": 0,
                "avg_in_degree": 0.0,
                "avg_out_degree": 0.0,
            }
        undirected = self.graph.to_undirected()
        weakly_connected = list(nx.weakly_connected_components(self.graph))
        return {
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "n_components": len(weakly_connected),
            "largest_component_size": max(len(c) for c in weakly_connected),
            "avg_in_degree": sum(d for _, d in self.graph.in_degree()) / self.graph.number_of_nodes(),
            "avg_out_degree": sum(d for _, d in self.graph.out_degree()) / self.graph.number_of_nodes(),
        }

    # ------------------------------------------------------------------
    # Personalised PageRank
    # ------------------------------------------------------------------

    def personalized_pagerank(
        self,
        seed_accounts: list[str],
        alpha: float = 0.85,
        max_iter: int = 100,
        tolerance: float = 1e-6,
    ) -> dict[str, float]:
        """Compute Personalised PageRank biased toward seed accounts.

        Seeds are accounts known to be high-risk (e.g. flagged by the
        ML detector).  PPR propagates their risk through the transaction
        network so that accounts connected to them also receive elevated
        scores.

        Parameters
        ----------
        seed_accounts : list[str]
            Account IDs to bias the random walk toward.
        alpha : float
            Damping factor (default 0.85).
        max_iter : int
            Maximum PageRank iterations.
        tolerance : float
            Convergence threshold.

        Returns
        -------
        dict[str, float]
            ``{account_id: pagerank_score}`` for every graph node.
            Higher = more risk exposure from the seeds.
        """
        if not seed_accounts or self.graph.number_of_nodes() == 0:
            return {}

        # Build personalisation vector
        valid_seeds = [s for s in seed_accounts if s in self.graph]
        if not valid_seeds:
            return {}

        personalization = {s: 1.0 / len(valid_seeds) for s in valid_seeds}

        try:
            pr = nx.pagerank(
                self.graph,
                alpha=alpha,
                personalization=personalization,
                max_iter=max_iter,
                tol=tolerance,
            )
        except nx.PowerIterationFailedConvergence:
            pr = nx.pagerank(
                self.graph,
                alpha=alpha,
                personalization=personalization,
                max_iter=max_iter * 2,
                tol=1e-3,
            )

        return pr

    # ------------------------------------------------------------------
    # Chain tracing (layering detection)
    # ------------------------------------------------------------------

    def trace_chains(
        self,
        source: str,
        max_hops: int = 4,
        time_window_hours: int = 72,
        min_amount: float = 0.0,
    ) -> list[list[dict[str, Any]]]:
        """Find multi-hop chains originating from *source*.

        Each chain is a sequence of accounts where consecutive
        transactions occur within *time_window_hours*.  This is the
        core "human would've missed this" demo feature for layering.

        Parameters
        ----------
        source : str
            Starting account ID.
        max_hops : int
            Maximum number of hops (edges) in a chain.
        time_window_hours : int
            Max hours allowed between consecutive transactions in a
            chain.
        min_amount : float
            Filter out edges below this amount.

        Returns
        -------
        list[list[dict]]
            Each inner list is a chain of dicts:
            ``[{"sender": ..., "receiver": ..., "amount": ...,
            "timestamp": ...}, ...]``.  Empty if no chains found.
        """
        source = str(source)
        if source not in self.graph or self._time_ordered is None:
            return []

        # Cached fast adjacency lookup
        edges_by_sender = self._build_edge_index(min_amount=min_amount)

        chains: list[list[dict]] = []

        def _dfs(current: str, current_chain: list[dict], depth: int) -> None:
            if depth >= max_hops:
                return
            outgoing = edges_by_sender.get(current, [])
            last_ts = current_chain[-1]["timestamp"] if current_chain else None

            for edge in outgoing:
                if last_ts is not None and pd.notnull(edge["timestamp"]) and pd.notnull(last_ts):
                    delta_h = (edge["timestamp"] - last_ts).total_seconds() / 3600
                    if delta_h > time_window_hours or delta_h < 0:
                        continue

                new_chain = current_chain + [edge]
                chains.append(new_chain)
                _dfs(edge["receiver"], new_chain, depth + 1)

        _dfs(source, [], 0)

        # Sort by length descending (longest chains first)
        chains.sort(key=len, reverse=True)
        return chains

    # ------------------------------------------------------------------
    # Subgraph extraction
    # ------------------------------------------------------------------

    def get_account_neighborhood(
        self, account_id: str, depth: int = 2
    ) -> nx.DiGraph:
        """Extract the subgraph within *depth* hops of *account_id*."""
        account_id = str(account_id)
        if account_id not in self.graph:
            return nx.DiGraph()

        nodes = {account_id}
        frontier = {account_id}
        for _ in range(depth):
            neighbors = set()
            for n in frontier:
                neighbors |= set(self.graph.successors(n))
                neighbors |= set(self.graph.predecessors(n))
            nodes |= neighbors
            frontier = neighbors - nodes
            if not frontier:
                break

        return self.graph.subgraph(nodes).copy()

    # ------------------------------------------------------------------
    # Export for PyVis
    # ------------------------------------------------------------------

    def export_for_pyvis(
        self,
        risk_scores: dict[str, float] | None = None,
        pagerank_scores: dict[str, float] | None = None,
        focus_account: str | None = None,
        depth: int = 2,
        max_nodes: int = 100,
    ) -> dict[str, Any]:
        """Export node and edge data for PyVis visualisation.

        Parameters
        ----------
        risk_scores : dict or None
            Account -> ML anomaly score [0, 1].
        pagerank_scores : dict or None
            Account -> PPR score.
        focus_account : str or None
            Specific account to trace neighborhood for.
        depth : int
            Neighborhood depth if focus_account is set.
        max_nodes : int
            Cap maximum nodes returned when no focus account is provided
            to prevent browser DOM overload and JS freeze.

        Returns
        -------
        dict with ``nodes`` and ``edges`` lists.
        """
        if focus_account:
            subgraph = self.get_account_neighborhood(focus_account, depth=depth)
        else:
            all_nodes = list(self.graph.nodes())
            if len(all_nodes) > max_nodes:
                # Rank nodes by risk score + PPR score to pick top seed accounts
                def _priority(n: str) -> float:
                    r = risk_scores.get(n, 0.0) if risk_scores else 0.0
                    p = pagerank_scores.get(n, 0.0) if pagerank_scores else 0.0
                    return r * 100.0 + p

                ranked_seeds = sorted(all_nodes, key=_priority, reverse=True)
                included_nodes: set[str] = set()
                for seed in ranked_seeds:
                    if len(included_nodes) >= max_nodes:
                        break
                    # Include seed account along with its immediate connected transaction neighbors
                    neighborhood = set(self.graph.successors(seed)) | set(self.graph.predecessors(seed)) | {seed}
                    included_nodes.update(neighborhood)

                subgraph = self.graph.subgraph(included_nodes).copy()
            else:
                subgraph = self.graph

        # Determine node colors and sizes
        nodes_out = []
        for node in subgraph.nodes():
            risk = risk_scores.get(node, 0.0) if risk_scores else 0.0
            pr = pagerank_scores.get(node, 0.0) if pagerank_scores else 0.0

            if risk >= 0.8:
                color = "#e74c3c"  # red
            elif risk >= 0.5:
                color = "#f39c12"  # orange
            elif risk >= 0.2:
                color = "#f1c40f"  # yellow
            else:
                color = "#2ecc71"  # green

            size = 10 + pr * 500

            nodes_out.append({
                "id": str(node),
                "label": str(node),
                "color": color,
                "size": size,
                "title": f"Account: {node}\nRisk: {risk:.3f}\nPPR: {pr:.6f}",
            })

        # Edges
        edges_out = []
        for u, v, data in subgraph.edges(data=True):
            edges_out.append({
                "from": str(u),
                "to": str(v),
                "label": f"${data.get('weight', 0):,.0f}",
                "title": (
                    f"Total: ${data.get('weight', 0):,.0f}\n"
                    f"Txns: {data.get('tx_count', 1)}"
                ),
            })

        return {"nodes": nodes_out, "edges": edges_out}

    # ------------------------------------------------------------------
    # Get high-risk seeds from scores
    # ------------------------------------------------------------------

    def get_risk_seeds(
        self,
        scores: np.ndarray,
        accounts: pd.Series,
        top_n: int = 10,
        min_score: float = 0.7,
    ) -> list[str]:
        """Extract top-risk accounts from an anomaly score array."""
        acc_values = accounts.astype(str).values
        mask = scores >= min_score
        flagged = acc_values[mask]
        
        seen: set[str] = set()
        unique = []
        for acc in flagged:
            if acc not in seen:
                seen.add(acc)
                unique.append(acc)
        if len(unique) < top_n:
            # Fallback to top scored accounts if few exceed threshold
            top_idx = np.argsort(scores)[-top_n:][::-1]
            for idx in top_idx:
                acc = str(acc_values[idx])
                if acc not in seen:
                    seen.add(acc)
                    unique.append(acc)
        return unique[:top_n]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data" / "SAML-D.csv"
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        sys.exit(1)

    print("Loading SAML-D (50 000 rows sample)...")
    df = pd.read_csv(data_path, nrows=50_000)
    print(f"Loaded {len(df)} rows.")

    graph = TransactionGraph(df)
    graph.build()
    print(f"\nGraph stats: {graph.stats}")

    # PPR with sample seeds
    seed_candidates = [str(a) for a in df["Sender_account"].unique()[:5]]
    print(f"\nPPR with seeds: {seed_candidates}")
    ppr = graph.personalized_pagerank(seed_candidates)
    print(f"PPR computed for {len(ppr)} nodes.")
    top_ppr = sorted(ppr.items(), key=lambda x: x[1], reverse=True)[:5]
    for acc, score in top_ppr:
        print(f"  {acc}: PPR={score:.6f}")

    # Chain tracing
    source = seed_candidates[0]
    print(f"\nChain tracing from {source} (max 3 hops, 72 h window):")
    chains = graph.trace_chains(source, max_hops=3)
    if chains:
        print(f"  Found {len(chains)} chains, longest: {len(chains[0])} hops")
        for chain in chains[:3]:
            path = " -> ".join(
                f"${e['amount']:,.0f}" for e in chain
            )
            print(f"  Chain: {path}")
    else:
        print("  No chains found from this source (expected on first 50K rows).")
