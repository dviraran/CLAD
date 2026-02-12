"""
CPT code pricing lookup from Cleveland Clinic / Anthem data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class PriceInfo:
    """Pricing information for a CPT code."""
    cpt_code: str
    description: str
    negotiated_dollar: float | None
    min_charge: float | None
    max_charge: float | None
    plan_name: str

    def __bool__(self) -> bool:
        """True if any price is available."""
        return any([
            self.negotiated_dollar is not None,
            self.min_charge is not None,
            self.max_charge is not None,
        ])


class CPTPricingDatabase:
    """
    Look up CPT code prices from Cleveland Clinic / Anthem pricing data.

    Default plan is Managed Medicaid (most complete coverage in dataset).
    """

    DEFAULT_PRICES_PATH = Path("data/cpt/prices_clevelandclinic_anthem.csv")
    DEFAULT_PLAN = "Managed Medicaid"

    def __init__(
        self,
        prices_path: Path | str | None = None,
        default_plan: str = DEFAULT_PLAN,
    ):
        """
        Initialize pricing database.

        Args:
            prices_path: Path to prices CSV file
            default_plan: Default insurance plan to use
        """
        self.prices_path = Path(prices_path or self.DEFAULT_PRICES_PATH)
        self.default_plan = default_plan

        # State (loaded lazily)
        self._prices_df: pd.DataFrame | None = None
        self._code_to_prices: dict[str, dict[str, PriceInfo]] | None = None

    def _load_data(self) -> None:
        """Load and index pricing data."""
        if self._prices_df is not None:
            return

        if not self.prices_path.exists():
            raise FileNotFoundError(
                f"Pricing data not found: {self.prices_path}"
            )

        # Load CSV
        self._prices_df = pd.read_csv(self.prices_path, dtype={
            "primary_code": str,
            "description": str,
            "plan_name": str,
            "negotiated_dollar": float,
            "min_charge": float,
            "max_charge": float,
        })

        # Build lookup index: code -> plan -> PriceInfo
        self._code_to_prices = {}

        for _, row in self._prices_df.iterrows():
            code = str(row.get("primary_code", "")).strip()
            if not code:
                continue

            plan = str(row.get("plan_name", "")).strip()
            if not plan:
                continue

            if code not in self._code_to_prices:
                self._code_to_prices[code] = {}

            # Parse prices (handle NA/NaN)
            def safe_float(val) -> float | None:
                try:
                    if pd.isna(val):
                        return None
                    return float(val)
                except (ValueError, TypeError):
                    return None

            self._code_to_prices[code][plan] = PriceInfo(
                cpt_code=code,
                description=str(row.get("description", "")),
                negotiated_dollar=safe_float(row.get("negotiated_dollar")),
                min_charge=safe_float(row.get("min_charge")),
                max_charge=safe_float(row.get("max_charge")),
                plan_name=plan,
            )

    def get_price(
        self,
        cpt_code: str,
        plan: str | None = None,
    ) -> PriceInfo | None:
        """
        Get pricing for a CPT code.

        Args:
            cpt_code: The CPT code to look up
            plan: Insurance plan (default: Managed Medicaid)

        Returns:
            PriceInfo or None if not found
        """
        self._load_data()

        plan = plan or self.default_plan
        code = str(cpt_code).strip()

        if code not in self._code_to_prices:
            return None

        code_prices = self._code_to_prices[code]

        # Try requested plan first
        if plan in code_prices:
            return code_prices[plan]

        # Fall back to any available plan
        if code_prices:
            return next(iter(code_prices.values()))

        return None

    def get_all_plans(self, cpt_code: str) -> dict[str, PriceInfo]:
        """
        Get pricing across all available insurance plans.

        Args:
            cpt_code: The CPT code to look up

        Returns:
            Dict mapping plan name to PriceInfo
        """
        self._load_data()

        code = str(cpt_code).strip()
        return self._code_to_prices.get(code, {})

    def has_code(self, cpt_code: str) -> bool:
        """Check if a CPT code has pricing data."""
        self._load_data()
        return str(cpt_code).strip() in self._code_to_prices

    @property
    def available_plans(self) -> list[str]:
        """List of all available insurance plans in the data."""
        self._load_data()
        plans = set()
        for code_prices in self._code_to_prices.values():
            plans.update(code_prices.keys())
        return sorted(plans)

    @property
    def num_codes(self) -> int:
        """Number of unique CPT codes with pricing."""
        self._load_data()
        return len(self._code_to_prices)

    def get_stats(self) -> dict:
        """Get summary statistics about the pricing data."""
        self._load_data()

        all_prices = []
        for code_prices in self._code_to_prices.values():
            for price_info in code_prices.values():
                if price_info.negotiated_dollar is not None:
                    all_prices.append(price_info.negotiated_dollar)

        if not all_prices:
            return {
                "num_codes": self.num_codes,
                "num_prices": 0,
                "plans": self.available_plans,
            }

        import numpy as np
        return {
            "num_codes": self.num_codes,
            "num_prices": len(all_prices),
            "plans": self.available_plans,
            "price_min": min(all_prices),
            "price_max": max(all_prices),
            "price_median": float(np.median(all_prices)),
            "price_mean": float(np.mean(all_prices)),
        }
