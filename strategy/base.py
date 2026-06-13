"""
Abstract base class for all strategies.
Layer: strategy. No DB/Redis/order imports.
"""

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """
    All strategy subclasses must implement generate_signal().
    Config is passed fresh each call so hot-reload works.
    """

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, config: dict) -> str | None:
        """
        Args:
            df: Feature DataFrame sorted ascending, all indicators computed.
                Last row = most recent closed candle.
            config: Full config dict (strategy section).

        Returns:
            'long' | 'short' | None
        """
        ...
