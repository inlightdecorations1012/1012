
"""
Feature registry constants for Forex Parquet Converter v13.4.
============================================================

v13.2-rc2: Split into BASE and FUTURE features for clarity.
"""

# Features we actually generate today (55 features)
BASE_FEATURES = [
    # Core Features (12)
    'close_bid', 'close_ask', 'high_bid', 'high_ask',
    'low_bid', 'low_ask', 'open_bid', 'open_ask',
    'mid_price_close', 'mid_price_high', 'mid_price_low', 'mid_price_open',

    # Basic Technical (11)
    'price_change', 'price_change_abs', 'ma_20', 'ma_50',
    'price_vs_ma_20', 'price_vs_ma_50', 'rsi_14',
    'atr', 'atr_pct', 'high_low_range', 'true_range',

    # Basic Microstructure (6)
    'bid_ask_spread', 'spread_pct', 'volume_bid', 'volume_ask',
    'volume_imbalance', 'has_volume_data',

    # Minimal Temporal (11)
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'is_london', 'is_ny', 'is_london_ny_overlap',
    'london_open', 'ny_open', 'london_close', 'ny_close',

    # Single Volatility (1)
    'volatility_20',

    # Transformer-Specific (6)
    'sequence_position', 'log_returns', 'is_extreme_move',
    'attention_hint_lookback_20', 'time_since_last_extreme',
    'cumulative_return_100',

    # Price Action/Additional (8)
    'bar_size_vs_atr', 'wick_ratio_upper', 'wick_ratio_lower',
    'macd_signal_cross', 'bb_width', 'bb_position',
    'price_acceleration', 'days_from_start',
]

# Placeholder features for future implementation (22 features)
FUTURE_FEATURES = [
    # Future Market Structure (8)
    'fvg_above', 'fvg_below', 'nearest_resistance', 'nearest_support',
    'is_at_key_level', 'pivot_position', 'market_regime', 'adx_value',

    # Future Macro Context (7)
    'yield_spread_10y', 'yield_spread_change', 'gold_pct_change',
    'gold_momentum', 'sp500_pct_change', 'risk_sentiment', 'macro_alignment',

    # Future Multi-Timeframe (5)
    'trend_h1', 'trend_h4', 'trend_d1', 'trend_alignment', 'trend_strength',

    # Additional features to reach 77 (2)
    'order_flow_imbalance', 'liquidity_score'
]

# Complete list of all 77 features
COMPLETE_77_FEATURES = BASE_FEATURES + FUTURE_FEATURES

# Verify counts
assert len(BASE_FEATURES) == 55, f"Expected 55 base features, got {len(BASE_FEATURES)}"
assert len(FUTURE_FEATURES) == 22, f"Expected 22 future features, got {len(FUTURE_FEATURES)}"
assert len(COMPLETE_77_FEATURES) == 77, f"Expected 77 total features, got {len(COMPLETE_77_FEATURES)}"
