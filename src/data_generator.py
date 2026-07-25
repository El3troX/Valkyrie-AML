"""
Synthetic SAML-D dataset generator for Valkyrie-AML.

Generates realistic labeled transaction data with four typologies:
normal, structuring, smurfing, and layering. Schema matches the SAML-D
dataset format used in anti-money-laundering research.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CURRENCIES = ["USD", "EUR", "GBP"]
BANK_LOCATIONS = [
    "US", "GB", "DE", "FR", "CH", "SG", "HK", "AE", "KY", "PA",
    "LU", "NL", "IE", "JP", "AU",
]
PAYMENT_TYPES = ["wire", "ach", "cash", "crypto", "check"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class SAMLDataGenerator:
    """Generate synthetic SAML-D-compliant AML training data.

    Produces a DataFrame with the SAML-D schema plus ground-truth labels
    ``is_suspicious`` (0/1) and ``typology`` (normal | structuring |
    smurfing | layering).

    Parameters
    ----------
    n_transactions : int
        Total number of transactions to generate.
    suspicious_ratio : float
        Approximate fraction of transactions that are suspicious.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_transactions: int = 10_000,
        suspicious_ratio: float = 0.20,
        seed: int = 42,
    ) -> None:
        self.n_transactions = n_transactions
        self.suspicious_ratio = suspicious_ratio
        self.rng = np.random.default_rng(seed)
        self.accounts = [f"ACC-{i:04d}" for i in range(1, 501)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, save: bool = True) -> pd.DataFrame:
        """Generate the full synthetic dataset.

        Returns
        -------
        pd.DataFrame
            DataFrame with SAML-D columns plus ``is_suspicious`` and
            ``typology``.
        """
        n_suspicious = int(self.n_transactions * self.suspicious_ratio)
        n_normal = self.n_transactions - n_suspicious

        # Split suspicious roughly equally across typologies
        n_each = n_suspicious // 3
        remainder = n_suspicious - n_each * 3

        frames: list[pd.DataFrame] = []

        # Normal transactions
        frames.append(self._generate_normal(n_normal))

        # Structuring
        frames.append(self._generate_structuring(n_each + remainder))

        # Smurfing
        frames.append(self._generate_smurfing(n_each))

        # Layering
        frames.append(self._generate_layering(n_each))

        df = pd.concat(frames, ignore_index=True)

        # Shuffle
        df = df.sample(frac=1, random_state=self.rng).reset_index(drop=True)

        # Add Transaction_ID
        df.insert(0, "Transaction_ID", [f"TXN-{i:05d}" for i in range(len(df))])

        if save:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            out_path = DATA_DIR / "synthetic_transactions.csv"
            df.to_csv(out_path, index=False)
            print(f"Saved {len(df)} transactions to {out_path}")

        self._validate(df)
        return df

    # ------------------------------------------------------------------
    # Internal generators
    # ------------------------------------------------------------------

    def _random_dates(self, n: int, start: str = "2024-01-01", end: str = "2024-12-31") -> pd.DataFrame:
        """Generate random dates and times within a range."""
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        days_range = (end_ts - start_ts).days
        days = self.rng.integers(0, days_range, size=n)
        dates = [start_ts + pd.Timedelta(days=int(d)) for d in days]

        # Business hours clustering: 70% between 9am-6pm
        hours = []
        for _ in range(n):
            if self.rng.random() < 0.70:
                h = self.rng.integers(9, 18)
            else:
                h = self.rng.integers(0, 24)
            m = self.rng.integers(0, 60)
            s = self.rng.integers(0, 60)
            hours.append(f"{h:02d}:{m:02d}:{s:02d}")

        return pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in dates],
            "Time": hours,
        })

    def _make_base(self, n: int) -> pd.DataFrame:
        """Create base DataFrame with random account pairs and metadata."""
        senders = self.rng.choice(self.accounts, size=n)
        # Ensure receiver != sender
        receivers = []
        for s in senders:
            pool = [a for a in self.accounts if a != s]
            receivers.append(self.rng.choice(pool))
        receivers = np.array(receivers)

        dt = self._random_dates(n)
        currencies = self.rng.choice(CURRENCIES, size=n, p=[0.6, 0.25, 0.15])
        locations = self.rng.choice(BANK_LOCATIONS, size=n)

        return pd.DataFrame({
            "Time": dt["Time"].values,
            "Date": dt["Date"].values,
            "Sender_account": senders,
            "Receiver_account": receivers,
            "Payment_currency": currencies,
            "Received_currency": currencies,  # same currency by default
            "Sender_bank_location": locations,
            "Receiver_bank_location": locations,
            "Payment_type": self.rng.choice(PAYMENT_TYPES, size=n, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        })

    def _generate_normal(self, n: int) -> pd.DataFrame:
        """Generate normal (non-suspicious) transactions."""
        df = self._make_base(n)

        # Log-normal amounts: mean ~$2,000, long tail up to ~$50K
        amounts = self.rng.lognormal(mean=np.log(2000), sigma=1.2, size=n)
        amounts = np.clip(amounts, 10, 100_000)
        df["Amount"] = np.round(amounts, 2)

        # ~10% cross-currency
        cross = self.rng.random(n) < 0.10
        df.loc[cross, "Received_currency"] = self.rng.choice(
            CURRENCIES, size=cross.sum()
        )

        df["is_suspicious"] = 0
        df["typology"] = "normal"
        return df

    def _generate_structuring(self, n: int) -> pd.DataFrame:
        """Generate structuring transactions — amounts just under $10,000.

        Pattern: multiple transactions from a sender in the $9,000–$9,999
        range within a short time window, designed to avoid CTR filing
        thresholds.
        """
        # Pick sender-receiver pairs that repeat
        n_groups = max(1, n // 8)  # ~8 txns per structuring cluster
        group_sizes = self.rng.multinomial(n, [1.0 / n_groups] * n_groups)

        frames: list[pd.DataFrame] = []
        for size in group_sizes:
            if size == 0:
                continue
            sender = self.rng.choice(self.accounts)
            pool = [a for a in self.accounts if a != sender]
            receiver = self.rng.choice(pool)

            dt_start = pd.Timestamp("2024-01-01") + pd.Timedelta(
                days=int(self.rng.integers(0, 300))
            )

            rows = []
            for j in range(size):
                # Amounts clustered just under $10K: $9,000 – $9,999
                amount = self.rng.uniform(9_000, 9_999)
                # Time: spread across 3-5 days, multiple per day
                offset_hours = int(self.rng.integers(0, 120))
                tx_time = dt_start + pd.Timedelta(hours=offset_hours)
                h = tx_time.hour
                m = tx_time.minute
                s = tx_time.second
                curr = self.rng.choice(CURRENCIES, p=[0.7, 0.2, 0.1])
                loc = self.rng.choice(BANK_LOCATIONS, p=[0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])

                rows.append({
                    "Time": f"{h:02d}:{m:02d}:{s:02d}",
                    "Date": tx_time.strftime("%Y-%m-%d"),
                    "Sender_account": sender,
                    "Receiver_account": receiver,
                    "Amount": round(amount, 2),
                    "Payment_currency": curr,
                    "Received_currency": curr,
                    "Sender_bank_location": loc,
                    "Receiver_bank_location": self.rng.choice(BANK_LOCATIONS),
                    "Payment_type": "wire",
                })
            frames.append(pd.DataFrame(rows))

        df = pd.concat(frames, ignore_index=True).head(n)
        df["is_suspicious"] = 1
        df["typology"] = "structuring"
        return df

    def _generate_smurfing(self, n: int) -> pd.DataFrame:
        """Generate smurfing transactions — many small amounts from
        multiple senders funneling into one receiver.

        Pattern: a single receiver collects funds from 10-30 different
        senders, each contributing $500–$2,500.
        """
        receiver = self.rng.choice(self.accounts)
        n_senders = min(self.rng.integers(10, 31), n)
        senders = self.rng.choice(
            [a for a in self.accounts if a != receiver],
            size=n_senders,
            replace=False,
        )

        # Distribute transactions across senders
        amounts_per_sender = self.rng.multinomial(n, [1.0 / n_senders] * n_senders)

        frames: list[pd.DataFrame] = []
        dt_start = pd.Timestamp("2024-01-01") + pd.Timedelta(
            days=int(self.rng.integers(0, 300))
        )

        for sender, count in zip(senders, amounts_per_sender):
            if count == 0:
                continue
            for _ in range(count):
                amount = self.rng.uniform(500, 2_500)
                offset_hours = int(self.rng.integers(0, 168))  # within 7 days
                tx_time = dt_start + pd.Timedelta(hours=offset_hours)
                curr = self.rng.choice(CURRENCIES, p=[0.6, 0.25, 0.15])

                frames.append(pd.DataFrame([{
                    "Time": f"{tx_time.hour:02d}:{tx_time.minute:02d}:{tx_time.second:02d}",
                    "Date": tx_time.strftime("%Y-%m-%d"),
                    "Sender_account": sender,
                    "Receiver_account": receiver,
                    "Amount": round(amount, 2),
                    "Payment_currency": curr,
                    "Received_currency": curr,
                    "Sender_bank_location": self.rng.choice(BANK_LOCATIONS),
                    "Receiver_bank_location": "KY",  # Cayman Islands
                    "Payment_type": self.rng.choice(["cash", "wire", "ach"], p=[0.4, 0.35, 0.25]),
                }]))

        df = pd.concat(frames, ignore_index=True).head(n)
        df["is_suspicious"] = 1
        df["typology"] = "smurfing"
        return df

    def _generate_layering(self, n: int) -> pd.DataFrame:
        """Generate layering transactions — multi-hop chains designed
        to obscure the origin of funds.

        Pattern: A -> B -> C -> D -> E with amounts decreasing 5-15%
        at each hop, all within 24-72 hours.
        """
        chain_len = self.rng.integers(3, 6)  # 3-5 hops per chain
        n_chains = max(1, n // chain_len)

        frames: list[pd.DataFrame] = []
        dt_start = pd.Timestamp("2024-01-01") + pd.Timedelta(
            days=int(self.rng.integers(0, 300))
        )

        for _ in range(n_chains):
            # Create a chain of unique accounts
            chain_accounts = list(
                self.rng.choice(self.accounts, size=chain_len + 1, replace=False)
            )
            initial_amount = self.rng.uniform(15_000, 80_000)
            current_amount = initial_amount
            chain_start = dt_start + pd.Timedelta(hours=int(self.rng.integers(0, 600)))

            for i in range(chain_len):
                sender = chain_accounts[i]
                receiver = chain_accounts[i + 1]
                # Amount decreases 5-15% at each hop (fees)
                fee_pct = self.rng.uniform(0.05, 0.15)
                current_amount *= (1 - fee_pct)

                # Each hop happens 2-24 hours after the previous
                hop_offset = int(self.rng.integers(2, 24))
                tx_time = chain_start + pd.Timedelta(hours=hop_offset * (i + 1))
                curr = self.rng.choice(CURRENCIES, p=[0.65, 0.25, 0.10])

                frames.append(pd.DataFrame([{
                    "Time": f"{tx_time.hour:02d}:{tx_time.minute:02d}:{tx_time.second:02d}",
                    "Date": tx_time.strftime("%Y-%m-%d"),
                    "Sender_account": sender,
                    "Receiver_account": receiver,
                    "Amount": round(current_amount, 2),
                    "Payment_currency": curr,
                    "Received_currency": curr,
                    "Sender_bank_location": self.rng.choice(BANK_LOCATIONS),
                    "Receiver_bank_location": self.rng.choice(BANK_LOCATIONS),
                    "Payment_type": self.rng.choice(["wire", "wire", "ach"]),
                }]))

        df = pd.concat(frames, ignore_index=True).head(n)
        df["is_suspicious"] = 1
        df["typology"] = "layering"
        return df

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        """Run basic sanity checks on the generated dataset."""
        assert len(df) > 0, "Empty DataFrame"
        assert df.isnull().sum().sum() == 0, "Null values found"

        required_cols = [
            "Transaction_ID", "Date", "Time", "Sender_account",
            "Receiver_account", "Amount", "Payment_currency",
            "Received_currency", "Sender_bank_location",
            "Receiver_bank_location", "Payment_type",
            "is_suspicious", "typology",
        ]
        missing = set(required_cols) - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

        # Check structuring amounts are in the $9K-$9.9K range
        struct_amounts = df.loc[df["typology"] == "structuring", "Amount"]
        if len(struct_amounts) > 0:
            assert struct_amounts.min() >= 9_000, (
                f"Structuring min amount {struct_amounts.min()} < 9000"
            )
            assert struct_amounts.max() <= 10_000, (
                f"Structuring max amount {struct_amounts.max()} > 10000"
            )

        print("✓ Validation passed")
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        print(f"  Typology distribution:\n{df['typology'].value_counts().to_string()}")
        print(f"  Accounts: {df['Sender_account'].nunique()} unique")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gen = SAMLDataGenerator(n_transactions=10_000, seed=42)
    df = gen.generate(save=True)
    print(f"\nSample rows:\n{df.head(10).to_string()}")
