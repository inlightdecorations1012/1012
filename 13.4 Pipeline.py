
"""
Pipeline stages and feature engineering for Forex Parquet Converter v13.4.
========================================================================

v13.4 Changes:
- DataValidationStage: Added preserve mode with validation masks
- DataValidationStage: Dynamic spread filter based on median spread
- DataValidationStage: Fixed price validation logic (numeric conversion first)
- DataValidationStage: Use configurable thresholds from config
- TargetGenerationStage: Dynamic ATR multipliers for better class balance
- TargetGenerationStage: Added horizon guard for short DataFrames
- FeatureEngineeringStage: Per-timeframe feature validation
- Enhanced logging for validation decisions (info vs warning)
- FeatureRegistry: Methods now return list of features created

Pipeline stages:
1. DataValidationStage - Validate and clean/flag input data
2. FeatureEngineeringStage - Generate features from raw market data
3. TargetGenerationStage - Create targets with proper lookahead prevention
4. NormalizationStage - Normalize features for model training
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
import logging
import re
from datetime import datetime
import hashlib
import pytz

# Try to import bottleneck for performance optimization
try:
    import bottleneck as bn
    BOTTLENECK_AVAILABLE = True
except ImportError:
    bn = None
    BOTTLENECK_AVAILABLE = False

from config import (
    TimeframeConfig, TargetConfig, NormalizationConfig,
    DataQualityError, QualityControlConfig
)
from utils import (
    safe_rolling_window, get_merge_tolerance, collect_window_sizes,
    ForexMarketCalendar, validate_dataframe_integrity, parse_horizon_from_column,
    FeatureCache, get_expected_features_for_timeframe, validate_timeframe_features
)

logger = logging.getLogger(__name__)

# Public API
__all__ = [
    'PipelineStage',
    'DataValidationStage',
    'FeatureEngineeringStage',
    'TargetGenerationStage',
    'NormalizationStage',
    'FeatureRegistry'
]


# === Abstract Base Class for Pipeline Stages ===

class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""

    @abstractmethod
    def transform(self, df: pd.DataFrame, config: TimeframeConfig) -> pd.DataFrame:
        """Transform the dataframe.

        Args:
            df: Input dataframe
            config: Timeframe configuration

        Returns:
            Transformed dataframe
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the stage name.

        Returns:
            Stage name
        """
        pass


# === Stage 1: Data Validation (Updated in v13.4) ===

class DataValidationStage(PipelineStage):
    """Validate and clean input data with preserve mode support."""

    @property
    def name(self) -> str:
        return "DataValidation"

    def transform(self, df: pd.DataFrame, config: TimeframeConfig) -> pd.DataFrame:
        """Validate and clean the input dataframe.

        v13.4: Added preserve mode with validation masks and dynamic spread filter.
               Fixed price validation logic.

        Args:
            df: Input dataframe
            config: Timeframe configuration

        Returns:
            Cleaned/flagged dataframe

        Raises:
            DataQualityError: If data quality issues found in drop mode
        """
        if df.empty:
            raise DataQualityError(f"Empty DataFrame for {config.name}")

        initial_rows = len(df)
        qc_config = config.quality_control_config

        # Handle timezone-aware timestamps properly
        if 'timestamp' in df.columns:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

            # v13.2-rc2: Fixed timezone handling to avoid double-localization
            if pd.api.types.is_datetime64tz_dtype(df['timestamp']):
                # Already has timezone
                current_tz = df['timestamp'].dt.tz
                target_tz = pytz.timezone(config.timezone)

                # Only convert if different timezone
                if current_tz != target_tz:
                    df['timestamp'] = df['timestamp'].dt.tz_convert(config.timezone)
                # else: already in correct timezone, no action needed
            else:
                # No timezone, localize first
                df['timestamp'] = df['timestamp'].dt.tz_localize(config.timezone)

            # Drop any rows where timestamp parsing failed
            df = df.dropna(subset=['timestamp'])

        # v13.4: Initialize validation masks in preserve mode
        if qc_config.validation_mode == 'preserve':
            df['is_valid_price'] = True
            df['is_valid_spread'] = True
            df['is_valid_time_gap'] = True

        # Calculate mid prices
        df = self._calculate_mid_prices(df)

               # 1. Price sanity checks  (v13.4-fix3)
        price_columns = [
            'close_bid', 'close_ask', 'high_bid', 'high_ask',
            'low_bid',   'low_ask',  'open_bid', 'open_ask'
        ]

        # Convert to numeric first …
        price_data = df[price_columns].apply(pd.to_numeric, errors='coerce')

        # … and write the numeric values back to the main frame
        df[price_columns] = price_data

        # Build the price-validity mask
        price_checks  = price_data.notna().all(axis=1)          # no NaNs
        price_checks &= (price_data > 0).all(axis=1)            # strictly > 0
        price_checks &= (price_data['high_bid'] >= price_data['low_bid'])
        price_checks &= (price_data['high_ask'] >= price_data['low_ask'])

        # Drop or flag, depending on QC mode
        if qc_config.validation_mode == 'drop':
            invalid_price_rows = (~price_checks).sum()
            if invalid_price_rows > 0:
                logger.warning(f"Dropping {invalid_price_rows} rows with invalid prices")
                df = df[price_checks]
        else:  # preserve mode
            df['is_valid_price'] = price_checks
            invalid_price_rows = (~price_checks).sum()
            if invalid_price_rows > 0:
                logger.info(f"Flagged {invalid_price_rows} rows with invalid prices")


        # 2. Bid-Ask spread validation
        df['bid_ask_spread'] = df['close_ask'] - df['close_bid']

        # v13.4: Dynamic spread threshold based on median
        median_spread = df[df['bid_ask_spread'] > 0]['bid_ask_spread'].median()
        dynamic_threshold = max(
            qc_config.spread_floor,  # v13.4: Use configurable floor
            qc_config.spread_multiplier * median_spread  # v13.4: Use configurable multiplier
        )

        spread_checks = (
            (df['bid_ask_spread'] >= qc_config.min_spread_pct) &
            (df['bid_ask_spread'] <= dynamic_threshold)
        )

        # Apply or flag based on mode
        if qc_config.validation_mode == 'drop':
            invalid_spread_rows = (~spread_checks).sum()
            if invalid_spread_rows > 0:
                logger.warning(
                    f"Dropping {invalid_spread_rows} rows with invalid spreads "
                    f"(threshold: {dynamic_threshold:.6f}, median: {median_spread:.6f})"
                )
                df = df[spread_checks]
        else:
            # Preserve mode
            df['is_valid_spread'] = spread_checks
            invalid_spread_rows = (~spread_checks).sum()
            if invalid_spread_rows > 0:
                logger.info(
                    f"Flagged {invalid_spread_rows} rows with invalid spreads "
                    f"(threshold: {dynamic_threshold:.6f}, median: {median_spread:.6f})"
                )

        # 3. Time gap validation
        if len(df) > 1:
            # Calculate expected frequency
            freq_map = {
                '15T': pd.Timedelta('15min'), '15min': pd.Timedelta('15min'),
                '1H': pd.Timedelta('1h'), '1h': pd.Timedelta('1h'),
                '4H': pd.Timedelta('4h'), '4h': pd.Timedelta('4h'),
                '1D': pd.Timedelta('1d'), '1d': pd.Timedelta('1d'),
                '1W': pd.Timedelta('7d'), '1w': pd.Timedelta('7d')
            }
            expected_freq = freq_map.get(config.frequency, pd.Timedelta('1min'))

            # Check gaps
            time_diffs = df['timestamp'].diff()
            max_allowed_gap = expected_freq * qc_config.max_time_gap_multiplier

            # Find gaps
            gap_mask = time_diffs > max_allowed_gap
            gap_positions = df.index[gap_mask].tolist()

            if gap_positions:
                # Check if gaps are expected (weekends/holidays)
                calendar_mode = getattr(tf_config, "_calendar_mode", "basic")

                # Retrieve or create the ForexMarketCalendar instance
                calendar = getattr(tf_config, "calendar", None)
                if calendar is None:
                    calendar = ForexMarketCalendar(calendar_mode=calendar_mode)

                unexpected_gaps = []
                for gap_pos in gap_positions:
                    if not self._is_expected_gap(df, gap_pos, calendar):
                        unexpected_gaps.append(gap_pos)

                if unexpected_gaps:
                    if qc_config.validation_mode == 'drop':
                        logger.warning(f"Found {len(unexpected_gaps)} unexpected time gaps")
                        # In drop mode, we might remove these rows or handle differently
                    else:
                        # Flag the rows after gaps
                        df.loc[unexpected_gaps, 'is_valid_time_gap'] = False
                        logger.info(f"Flagged {len(unexpected_gaps)} rows with unexpected time gaps")

        # 4. Remove duplicates (always remove, not flag)
        dup_count = df.duplicated(subset=['timestamp']).sum()
        if dup_count > 0:
            logger.info(f"Removing {dup_count} duplicate timestamps")
            df = df.drop_duplicates(subset=['timestamp'], keep='first')

        # Calculate final statistics
        final_rows = len(df)
        rows_removed_or_flagged = initial_rows - final_rows

        if qc_config.validation_mode == 'drop':
            if rows_removed_or_flagged > 0:
                logger.info(
                    f"[{config.name}] Data validation complete: "
                    f"{rows_removed_or_flagged}/{initial_rows} rows removed "
                    f"({rows_removed_or_flagged/initial_rows*100:.1f}%)"
                )
        else:
            # Count total flagged rows
            total_flagged = 0
            if 'is_valid_price' in df.columns:
                total_flagged = (~(df['is_valid_price'] &
                                 df['is_valid_spread'] &
                                 df['is_valid_time_gap'])).sum()

            logger.info(
                f"[{config.name}] Data validation complete: "
                f"{total_flagged}/{initial_rows} rows flagged "
                f"({total_flagged/initial_rows*100:.1f}%)"
            )

        # Final sanity check
        if final_rows < qc_config.min_rows_per_timeframe:
            raise DataQualityError(
                f"Insufficient data for {config.name}: {final_rows} rows "
                f"(minimum: {qc_config.min_rows_per_timeframe})"
            )

        # Validate dataframe integrity
        validate_dataframe_integrity(df, f"DataValidation-{config.name}")

        return df

    def _calculate_mid_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mid prices from bid/ask."""
        df['mid_price_close'] = (df['close_bid'] + df['close_ask']) / 2
        df['mid_price_open'] = (df['open_bid'] + df['open_ask']) / 2
        df['mid_price_high'] = (df['high_bid'] + df['high_ask']) / 2
        df['mid_price_low'] = (df['low_bid'] + df['low_ask']) / 2
        return df

    def _is_expected_gap(self, df: pd.DataFrame, gap_pos: int,
                        calendar: ForexMarketCalendar) -> bool:
        """Check if a time gap is expected (weekend/holiday).

        Args:
            df: DataFrame
            gap_pos: Position of gap
            calendar: Market calendar

        Returns:
            True if gap is expected
        """
        prev_time = df['timestamp'].iloc[gap_pos - 1]
        curr_time = df['timestamp'].iloc[gap_pos]
        days_between = (curr_time.date() - prev_time.date()).days

        # Weekend gap
        if prev_time.weekday() == 4 and curr_time.weekday() <= 1 and days_between <= 4:
            return True

        # Holiday gap
        if calendar.is_forex_closed(prev_time) or calendar.is_forex_closed(curr_time):
            return True

        return False


# === Stage 2: Feature Engineering (Updated in v13.4) ===

class FeatureEngineeringStage(PipelineStage):
    """Generate features from raw market data with caching support."""

    def __init__(self, feature_registry: 'FeatureRegistry',
                 feature_cache: Optional['FeatureCache'] = None):
        """Initialize with feature registry and optional cache.

        Args:
            feature_registry: Registry of feature functions
            feature_cache: Optional feature cache for incremental enrichment
        """
        self.feature_registry = feature_registry
        self.feature_cache = feature_cache

    @property
    def name(self) -> str:
        return "FeatureEngineering"

    def transform(self, df: pd.DataFrame, config: TimeframeConfig) -> pd.DataFrame:
        """Apply feature engineering with caching.

        v13.4: Added per-timeframe feature validation.

        Args:
            df: Input dataframe
            config: Timeframe configuration

        Returns:
            DataFrame with features added
        """
        initial_cols = set(df.columns)

        # Calculate base metrics first
        df = self._calculate_base_metrics(df, config)

        # Apply feature sets with caching
        for feature_name in config.enabled_features:
            if feature_name in self.feature_registry.features:
                # Generate cache key
                cache_key = self._generate_feature_cache_key(
                    df, config.name, feature_name, config.feature_params.get(feature_name, {})
                )

                # Try to get from cache
                if self.feature_cache is not None:
                    cached_features = self.feature_cache.get(cache_key)
                    if cached_features is not None:
                        # Apply cached features
                        for col_name, col_data in cached_features.items():
                            if len(col_data) == len(df):
                                df[col_name] = col_data
                        logger.debug(f"Applied cached features for {feature_name}")
                        continue

                # Compute features
                df_before = df.copy()
                df = self.feature_registry.apply_feature(df, feature_name, config.feature_params)

                # Cache new features
                if self.feature_cache is not None:
                    new_cols = set(df.columns) - set(df_before.columns)
                    if new_cols:
                        features_to_cache = {col: df[col].copy() for col in new_cols}
                        self.feature_cache.set(cache_key, features_to_cache)
                        logger.debug(f"Cached {len(new_cols)} features for {feature_name}")
            else:
                logger.warning(f"Unknown feature set: {feature_name}")

        # Add validity masks
        df = self._add_validity_masks(df, config)

        # Clean up any NaNs introduced
        df = self._handle_feature_nans(df)

        # v13.4: Validate features match expectations for this timeframe
        validation_result = validate_timeframe_features(df, config, self.feature_registry)

        if validation_result['missing']:
            logger.error(
                f"[{config.name}] Missing requested features: {validation_result['missing']}"
            )
        if validation_result['unexpected']:
            logger.warning(
                f"[{config.name}] Unexpected features found: {validation_result['unexpected']}"
            )

        logger.info(
            f"[{config.name}] Feature coverage: {validation_result['coverage_pct']:.1f}% "
            f"({validation_result['actual_count']}/{validation_result['expected_count']} features)"
        )

        # Store validation result for reporting
        df.attrs['feature_validation'] = validation_result

        # Validate integrity
        validate_dataframe_integrity(df, f"FeatureEngineering-{config.name}")

        num_new_features = len(set(df.columns) - initial_cols)
        logger.info(f"[{config.name}] Added {num_new_features} new features")

        return df

    def _generate_feature_cache_key(self, df: pd.DataFrame, timeframe: str,
                                  feature_name: str, params: Dict[str, Any]) -> str:
        """Generate cache key for feature set.

        v13.4: Include timeframe in cache key.

        Args:
            df: DataFrame (for data hash)
            timeframe: Timeframe name
            feature_name: Feature set name
            params: Feature parameters

        Returns:
            Cache key string
        """
        # Create a hash of the input data
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(df[['timestamp', 'mid_price_close']].head(100), index=False).values
        ).hexdigest()[:8]

        # Create a hash of parameters
        param_str = str(sorted(params.items()))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

        # v13.4: Include timeframe in key
        return f"{timeframe}_{feature_name}_{data_hash}_{param_hash}"

    def _calculate_base_metrics(self, df: pd.DataFrame, config: TimeframeConfig) -> pd.DataFrame:
        """Calculate base metrics used by multiple features.

        Args:
            df: Input dataframe
            config: Timeframe configuration

        Returns:
            DataFrame with base metrics added
        """
        # ATR calculation (used by many features)
        if 'atr' not in df.columns:
            high_low = df['mid_price_high'] - df['mid_price_low']
            high_close = (df['mid_price_high'] - df['mid_price_close'].shift(1)).abs()
            low_close = (df['mid_price_low'] - df['mid_price_close'].shift(1)).abs()

            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['true_range'] = tr

            # Use bottleneck if available
            atr_period = config.feature_params.get('volatility_basic', {}).get('atr_period', 14)
            if BOTTLENECK_AVAILABLE and len(df) > atr_period:
                atr_values = bn.move_mean(tr.values, window=atr_period, min_count=1)
                # Handle NaN values at the beginning
                atr_values[:atr_period-1] = tr.iloc[:atr_period-1].expanding(min_periods=1).mean().values
                df['atr'] = atr_values
            else:
                df['atr'] = tr.ewm(span=atr_period, adjust=False, min_periods=1).mean()

            # ATR percentage
            df['atr_pct'] = df['atr'] / df['mid_price_close']

        return df

    def _add_validity_masks(self, df: pd.DataFrame, config: TimeframeConfig) -> pd.DataFrame:
        """Add data validity masks for warm-up periods.

        v13.4: Enhanced to work with preserve mode validation masks.

        Args:
            df: DataFrame
            config: Timeframe configuration

        Returns:
            DataFrame with validity masks added
        """
        # If not already set by validation stage
        if 'data_validity_mask' not in df.columns:
            df['data_validity_mask'] = 1.0

        # Calculate maximum warm-up period needed
        warmup_periods = collect_window_sizes(config.feature_params)

        # Also consider multi-horizon targets
        if hasattr(config.target_config, 'multi_horizon'):
            warmup_periods.extend(config.target_config.multi_horizon)

        # Add normalization window
        warmup_periods.append(config.normalization_config.window)

        max_warmup = max(warmup_periods) if warmup_periods else 0

        if max_warmup > 0 and len(df) > max_warmup:
            # v13.4: Combine with existing validity mask
            warmup_mask = pd.Series(1.0, index=df.index)

            # Set mask to 0 during warm-up
            warmup_mask.iloc[:max_warmup] = 0.0

            # Gradual transition
            transition_period = min(10, max_warmup // 2)
            for i in range(transition_period):
                idx = max_warmup + i
                if idx < len(df):
                    warmup_mask.iloc[idx] = (i + 1) / transition_period

            # Combine with existing validity mask
            df['data_validity_mask'] = df['data_validity_mask'] * warmup_mask

            logger.info(f"[{config.name}] Added validity masks, warm-up period: {max_warmup}")

        return df

    def _handle_feature_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle any NaNs introduced during feature engineering.

        Args:
            df: DataFrame

        Returns:
            DataFrame with NaNs handled
        """
        # Check for NaNs
        nan_counts = df.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]

        if len(nan_cols) > 0:
            logger.debug(f"Found NaNs in {len(nan_cols)} columns after feature engineering")

            # For numeric columns, forward fill then backward fill
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            nan_numeric_cols = [col for col in nan_cols.index if col in numeric_cols]

            if nan_numeric_cols:
                df[nan_numeric_cols] = df[nan_numeric_cols].fillna(method='ffill').fillna(method='bfill')

            # Any remaining NaNs in numeric columns get filled with 0
            remaining_nans = df[numeric_cols].isna().sum()
            remaining_nan_cols = remaining_nans[remaining_nans > 0].index.tolist()
            if remaining_nan_cols:
                logger.debug(f"Filling remaining NaNs with 0 in: {remaining_nan_cols}")
                df[remaining_nan_cols] = df[remaining_nan_cols].fillna(0)

        return df


# === Stage 3: Target Generation (Updated in v13.4) ===

class TargetGenerationStage(PipelineStage):
    """Generate targets with proper lookahead bias prevention."""

    @property
    def name(self) -> str:
        return "TargetGeneration"

    def transform(self, df: pd.DataFrame, config: TimeframeConfig) -> pd.DataFrame:
        """Generate targets based on configuration.

        v13.4: Added dynamic ATR multipliers and horizon guard.

        Args:
            df: Input dataframe with features
            config: Timeframe configuration

        Returns:
            DataFrame with targets added
        """
        target_config = config.target_config

        if target_config.type == 'multi_horizon':
            df = self._generate_multi_horizon_targets(df, target_config, config)
        else:
            raise ValueError(f"Unknown target type: {target_config.type}")

        # Validate integrity
        validate_dataframe_integrity(df, f"TargetGeneration-{config.name}")

        return df

    def _generate_multi_horizon_targets(self, df: pd.DataFrame,
                                      target_config: TargetConfig,
                                      tf_config: TimeframeConfig) -> pd.DataFrame:
        """Generate multi-horizon targets with dynamic ATR multipliers.

        v13.4: Added dynamic ATR calculation and minority class targeting.

        Args:
            df: Input dataframe
            target_config: Target configuration
            tf_config: Timeframe configuration

        Returns:
            DataFrame with multi-horizon targets
        """
        close_price = df['mid_price_close']
        atr = df['atr']

        # v13.4: Dynamic ATR multiplier calculation
        if target_config.dynamic_atr_multiplier:
            logger.info(f"[{tf_config.name}] Calculating dynamic ATR multipliers...")
            optimal_multipliers = self._calculate_dynamic_atr_multipliers(
                df, target_config, tf_config.quality_control_config
            )
        else:
            # Use fixed multiplier for all horizons
            optimal_multipliers = {h: target_config.atr_multiplier for h in target_config.multi_horizon}

        # Generate targets for each horizon
        for horizon in target_config.multi_horizon:
            logger.info(f"[{tf_config.name}] Generating targets for horizon {horizon}")

            # Get multiplier for this horizon
            atr_mult = optimal_multipliers.get(horizon, target_config.atr_multiplier)

            # Calculate future returns
            future_returns = close_price.shift(-horizon) / close_price - 1

            # Normalize by ATR
            atr_normalized = future_returns / (atr * atr_mult).clip(lower=1e-8)

            # Create directional targets
            target_col = f'target_h{horizon}'
            df[target_col] = 0  # Hold
            df.loc[atr_normalized > 1, target_col] = 1   # Buy
            df.loc[atr_normalized < -1, target_col] = -1  # Sell

            # Soft labels if enabled
            if target_config.use_soft_labels:
                soft_col = f'target_h{horizon}_soft'
                df[soft_col] = df[target_col].copy()

                # Add soft labels for near-threshold cases
                soft_buy_mask = (atr_normalized > 0.7) & (atr_normalized <= 1.0)
                soft_sell_mask = (atr_normalized < -0.7) & (atr_normalized >= -1.0)

                df.loc[soft_buy_mask, soft_col] = 0.5   # Weak buy
                df.loc[soft_sell_mask, soft_col] = -0.5  # Weak sell

            # Mask invalid targets (lookahead prevention)
            mask_col = f'target_h{horizon}_mask'
            df[mask_col] = 1.0

            # v13.4: Guard against short DataFrames
            if len(df) > horizon:
                df.loc[df.index[-horizon:], mask_col] = 0.0
            else:
                # If DataFrame is too short, mask all targets
                df[mask_col] = 0.0
                logger.warning(
                    f"[{tf_config.name}] DataFrame too short for horizon {horizon} "
                    f"({len(df)} rows < {horizon} horizon)"
                )

            # Also mask targets during warm-up if validity mask exists
            if 'data_validity_mask' in df.columns:
                df[mask_col] = df[mask_col] * df['data_validity_mask']

            # Weight column for sample importance
            weight_col = f'target_h{horizon}_weight'
            df[weight_col] = df[mask_col].copy()

            # Log class distribution
            if mask_col in df.columns and df[mask_col].sum() > 0:
                valid_targets = df[df[mask_col] > 0][target_col]
                if len(valid_targets) > 0:
                    dist = valid_targets.value_counts(normalize=True).sort_index()
                    logger.info(
                        f"[{tf_config.name}] H{horizon} distribution (ATR mult={atr_mult:.2f}): "
                        f"Sell: {dist.get(-1, 0):.1%}, Hold: {dist.get(0, 0):.1%}, Buy: {dist.get(1, 0):.1%}"
                    )

        # Store optimal multipliers for reporting
        df.attrs['dynamic_atr_multipliers'] = optimal_multipliers

        return df

    def _calculate_dynamic_atr_multipliers(self, df: pd.DataFrame,
                                         target_config: TargetConfig,
                                         qc_config: QualityControlConfig) -> Dict[int, float]:
        """Calculate optimal ATR multipliers for each horizon.

        v13.4: New method to achieve target minority class percentage.

        Args:
            df: DataFrame with price and ATR data
            target_config: Target configuration
            qc_config: Quality control configuration

        Returns:
            Dictionary mapping horizon to optimal ATR multiplier
        """
        optimal_multipliers = {}
        target_pct = target_config.target_minority_class_pct

        close_price = df['mid_price_close']
        atr = df['atr']

        # Grid search parameters from config
        search_range = np.arange(
            qc_config.atr_grid_start,
            qc_config.atr_grid_stop,
            qc_config.atr_grid_step
        )

        for horizon in target_config.multi_horizon:
            # Skip if data too short
            if len(df) <= horizon:
                optimal_multipliers[horizon] = target_config.atr_multiplier
                continue

            # Calculate future returns
            future_returns = close_price.shift(-horizon) / close_price - 1

            # Only use valid data (not in tail)
            valid_mask = ~future_returns.isna()
            valid_returns = future_returns[valid_mask]
            valid_atr = atr[valid_mask]

            if len(valid_returns) == 0:
                optimal_multipliers[horizon] = target_config.atr_multiplier
                continue

            # Find multiplier that gives closest to target minority percentage
            best_mult = target_config.atr_multiplier
            best_diff = float('inf')

            for mult in search_range:
                # Calculate targets with this multiplier
                atr_normalized = valid_returns / (valid_atr * mult).clip(lower=1e-8)

                buy_pct = (atr_normalized > 1).sum() / len(atr_normalized)
                sell_pct = (atr_normalized < -1).sum() / len(atr_normalized)
                minority_pct = min(buy_pct, sell_pct)

                # Check if this is closer to target
                diff = abs(minority_pct - target_pct)
                if diff < best_diff:
                    best_diff = diff
                    best_mult = mult

                # Early stopping if close enough
                if diff < 0.001:  # Within 0.1% of target
                    break

            optimal_multipliers[horizon] = best_mult
            logger.debug(
                f"[{target_config.type}] H{horizon}: optimal ATR multiplier = {best_mult:.2f}"
            )

        return optimal_multipliers


# === Stage 4: Normalization ===

class NormalizationStage(PipelineStage):
    """Normalize features for model training."""

    @property
    def name(self) -> str:
        return "Normalization"

    def transform(self, df: pd.DataFrame, config: TimeframeConfig) -> pd.DataFrame:
        """Apply normalization to features.

        Args:
            df: Input dataframe
            config: Timeframe configuration

        Returns:
            DataFrame with normalized features added
        """
        norm_config = config.normalization_config

        # Get feature columns (exclude targets, metadata, and already normalized)
        feature_cols = [
            col for col in df.columns
            if not col.startswith('target_')
            and col not in ['timestamp', 'data_validity_mask']
            and not col.endswith('_norm')
            and col not in ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']  # Already normalized
            and not col.startswith('is_')  # Binary flags
            and pd.api.types.is_numeric_dtype(df[col])
        ]

        logger.info(f"[{config.name}] Normalizing {len(feature_cols)} features")

        # Store normalization statistics
        norm_stats = {}

        if norm_config.method == 'zscore':
            # Rolling z-score normalization
            for col in feature_cols:
                # Calculate rolling statistics
                roll_mean = safe_rolling_window(
                    df[col], norm_config.window, 'mean'
                )
                roll_std = safe_rolling_window(
                    df[col], norm_config.window, 'std'
                ).clip(lower=1e-8)

                # Normalize
                df[f'{col}_norm'] = (df[col] - roll_mean) / roll_std

                # Store final statistics for metadata
                norm_stats[col] = {
                    'method': 'rolling_zscore',
                    'window': norm_config.window,
                    'final_mean': float(roll_mean.iloc[-1]),
                    'final_std': float(roll_std.iloc[-1])
                }
        else:
            raise ValueError(f"Unknown normalization method: {norm_config.method}")

        # Store normalization statistics in attributes
        df.attrs['normalization_stats'] = norm_stats

        # Validate integrity
        validate_dataframe_integrity(df, f"Normalization-{config.name}")

        return df


# === Feature Registry (Updated in v13.4) ===

class FeatureRegistry:
    """Registry for feature engineering functions.

    v13.4: Feature methods now return list of features created.
    """

    def __init__(self):
        """Initialize feature registry."""
        self.features: Dict[str, Dict[str, Any]] = {}
        self._kwargs_cache: Dict[Tuple, Dict[str, Any]] = {}
        self._feature_lists: Dict[str, List[str]] = {}  # v13.4: Track features per set
        self._register_default_features()

    def register(self, name: str, func: Callable,
                params: Optional[Dict[str, Any]] = None,
                feature_list: Optional[List[str]] = None) -> None:
        """Register a feature function.

        v13.4: Added feature_list parameter.

        Args:
            name: Feature set name
            func: Feature function
            params: Default parameters
            feature_list: List of features created by this function
        """
        self.features[name] = {
            'function': func,
            'default_params': params or {}
        }
        if feature_list:
            self._feature_lists[name] = feature_list

    def get_features_for_set(self, name: str) -> List[str]:
        """Get list of features created by a feature set.

        v13.4: New method for dynamic feature discovery.

        Args:
            name: Feature set name

        Returns:
            List of feature names
        """
        return self._feature_lists.get(name, [])

    def apply_feature(
        self,
        df: pd.DataFrame,
        name: str,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """Apply a feature set to DataFrame with caching.

        Args:
            df: Input DataFrame
            name: Feature set name
            params: Parameters from config

        Returns:
            DataFrame with features added
        """
        # 1) Check that feature is registered
        if name not in self.features:
            raise ValueError(f"Unknown feature set: {name}")

        info = self.features[name]

        # 2) Extract parameters for this feature set
        raw_params: Dict[str, Any] = params.get(name, {})

        # 3) Convert lists to tuples for hashing
        norm_params = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in raw_params.items()
        }

        # 4) Build hashable cache key
        param_items = frozenset(norm_params.items())
        cache_key   = (name, param_items)

        # 5) Merge default + YAML parameters only first time
        if cache_key not in self._kwargs_cache:
            merged_kwargs = {**info["default_params"], **raw_params}
            self._kwargs_cache[cache_key] = merged_kwargs
        else:
            merged_kwargs = self._kwargs_cache[cache_key]

        # 6) Call the actual feature function
        return info["function"](df, **merged_kwargs)

    def _register_default_features(self) -> None:
        """Register all default feature sets."""
        # v13.4: Register with feature lists
        self.register('technical_basic', self._add_basic_technical_features,
                     {'ma_periods': [20, 50], 'rsi_period': 14},
                     feature_list=[
                         'price_change', 'price_change_abs', 'ma_20', 'ma_50',
                         'price_vs_ma_20', 'price_vs_ma_50', 'rsi_14',
                         'atr', 'atr_pct', 'high_low_range', 'true_range'
                     ])
        self.register('technical_advanced', self._add_advanced_technical_features,
                     {'bb_period': 20, 'bb_std': 2},
                     feature_list=['bb_width', 'bb_position', 'macd_signal_cross'])
        self.register('microstructure_basic', self._add_basic_microstructure_features,
                     feature_list=[
                         'bid_ask_spread', 'spread_pct', 'volume_bid', 'volume_ask',
                         'volume_imbalance', 'has_volume_data'
                     ])
        self.register('temporal_basic', self._add_basic_temporal_features,
                     feature_list=[
                         'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'days_from_start'
                     ])
        self.register('session_basic', self._add_basic_session_features,
                     feature_list=[
                         'is_london', 'is_ny', 'is_london_ny_overlap',
                         'london_open', 'ny_open', 'london_close', 'ny_close'
                     ])
        self.register('volatility_basic', self._add_basic_volatility_features,
                     {'window': 20}, feature_list=['volatility_20'])
        self.register('transformer_specific', self._add_transformer_specific_features,
                     feature_list=[
                         'sequence_position', 'log_returns', 'is_extreme_move',
                         'attention_hint_lookback_20', 'time_since_last_extreme',
                         'cumulative_return_100', 'price_acceleration'
                     ])
        self.register('attention_hints', self._add_attention_hint_features,
                     {'window': 20},
                     feature_list=[
                         'is_extreme_move', 'attention_hint_lookback_20', 'time_since_last_extreme'
                     ])
        self.register('price_action', self._add_price_action_features,
                     feature_list=['bar_size_vs_atr', 'wick_ratio_upper', 'wick_ratio_lower'])

    @staticmethod
    def _add_basic_technical_features(df: pd.DataFrame, ma_periods: List[int],
                                    rsi_period: int) -> pd.DataFrame:
        """Add basic technical indicators.

        v13.3: Fixed bottleneck NaN handling.
        Performance optimized with bottleneck where available.

        Args:
            df: Input dataframe
            ma_periods: List of MA periods
            rsi_period: RSI period

        Returns:
            DataFrame with features added
        """
        close = df['mid_price_close']

        # Price changes
        df['price_change'] = close.pct_change()
        df['price_change_abs'] = df['price_change'].abs()

        # Moving averages - optimized with bottleneck
        for period in ma_periods:
            if BOTTLENECK_AVAILABLE and len(df) > period:
                ma_values = bn.move_mean(close.values, window=period, min_count=1)
                # v13.3: Fix NaN handling - use expanding mean for initial values
                ma_values[:period-1] = close.iloc[:period-1].expanding(min_periods=1).mean().values
                df[f'ma_{period}'] = ma_values
            else:
                df[f'ma_{period}'] = safe_rolling_window(close, period, 'mean')

            # Price vs MA
            df[f'price_vs_ma_{period}'] = (close - df[f'ma_{period}']) / df[f'ma_{period}'].clip(lower=1e-8)

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=1).mean()

        rs = gain / loss.clip(lower=1e-8)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Handle edge cases
        df.loc[loss == 0, 'rsi_14'] = 100
        df['rsi_14'] = df['rsi_14'].fillna(50)

        # Range metrics
        df['high_low_range'] = (df['mid_price_high'] - df['mid_price_low']) / close

        return df

    @staticmethod
    def _add_advanced_technical_features(df: pd.DataFrame, bb_period: int,
                                       bb_std: float) -> pd.DataFrame:
        """Add advanced technical features.

        v13.3: Fixed bottleneck NaN handling.

        Args:
            df: Input dataframe
            bb_period: Bollinger Band period
            bb_std: Bollinger Band standard deviations

        Returns:
            DataFrame with features added
        """
        close = df['mid_price_close']

        # Bollinger Bands - optimized
        if BOTTLENECK_AVAILABLE and len(df) > bb_period:
            ma_values = bn.move_mean(close.values, window=bb_period, min_count=1)
            std_values = bn.move_std(close.values, window=bb_period, min_count=1)

            # v13.3: Fix NaN handling
            ma_values = np.where(np.isnan(ma_values),
                               close.expanding(min_periods=1).mean().values,
                               ma_values)
            std_values = np.where(np.isnan(std_values),
                                close.expanding(min_periods=1).std().values,
                                std_values)
            std_values = np.maximum(std_values, 1e-8)

            ma = pd.Series(ma_values, index=df.index)
            std = pd.Series(std_values, index=df.index)
        else:
            ma = safe_rolling_window(close, bb_period, 'mean')
            std = safe_rolling_window(close, bb_period, 'std')

        bb_upper = ma + (bb_std * std)
        bb_lower = ma - (bb_std * std)

        df['bb_width'] = (bb_upper - bb_lower) / ma.clip(lower=1e-8)
        bb_range = (bb_upper - bb_lower).clip(lower=1e-8)
        df['bb_position'] = ((close - bb_lower) / bb_range).clip(0, 1)

        # MACD with warm-up handling
        ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()

        # MACD signal cross
        macd_above = macd > macd_signal
        macd_cross = macd_above & (~macd_above.shift(1).fillna(False))
        df['macd_signal_cross'] = macd_cross.astype(int)

        return df

    @staticmethod
    def _add_basic_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add microstructure features - v13.2 aligned.

        Changed in v13.2: total_volume is temporary variable only.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with features added
        """
        # Spread metrics
        df['bid_ask_spread'] = df['close_ask'] - df['close_bid']
        df['spread_pct'] = df['bid_ask_spread'] / df['mid_price_close']

        # Volume features (if available)
        if 'volume_bid' in df.columns and 'volume_ask' in df.columns:
            # Keep individual volumes
            df['volume_bid'] = df['volume_bid'].fillna(0)
            df['volume_ask'] = df['volume_ask'].fillna(0)

            # Calculate total volume as temporary variable
            total_volume = df['volume_bid'] + df['volume_ask']

            # Volume imbalance
            df['volume_imbalance'] = np.where(
                total_volume > 0,
                (df['volume_bid'] - df['volume_ask']) / total_volume,
                0
            )

            # Binary flag for data availability
            df['has_volume_data'] = (total_volume > 0).astype(int)
        else:
            # No volume data
            df['volume_bid'] = 0
            df['volume_ask'] = 0
            df['volume_imbalance'] = 0
            df['has_volume_data'] = 0

        return df

    @staticmethod
    def _add_basic_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with features added
        """
        if 'timestamp' not in df.columns:
            return df

        # Extract time components
        hour = df['timestamp'].dt.hour
        dow = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

        # Days from start (for trend)
        df['days_from_start'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 86400

        # Note: hour and dow are temporary Series, not added to df

        return df

    @staticmethod
    def _add_basic_session_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add trading session features.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with features added
        """
        if 'timestamp' not in df.columns:
            return df

        # Get hour in UTC
        hour = df['timestamp'].dt.hour

        # Major session definitions (UTC)
        # London: 08:00-17:00 UTC
        df['is_london'] = ((hour >= 8) & (hour < 17)).astype(int)

        # New York: 13:00-22:00 UTC
        df['is_ny'] = ((hour >= 13) & (hour < 22)).astype(int)

        # Overlap: 13:00-17:00 UTC
        df['is_london_ny_overlap'] = (df['is_london'] & df['is_ny']).astype(int)

        # Session boundaries
        df['london_open'] = (hour == 8).astype(int)
        df['ny_open'] = (hour == 13).astype(int)
        df['london_close'] = (hour == 16).astype(int)
        df['ny_close'] = (hour == 21).astype(int)

        return df

    @staticmethod
    def _add_basic_volatility_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add volatility features.

        Args:
            df: Input dataframe
            window: Window for volatility calculation

        Returns:
            DataFrame with features added
        """
        # Return-based volatility
        returns = df['mid_price_close'].pct_change()

        if BOTTLENECK_AVAILABLE and len(df) > window:
            vol_values = bn.move_std(returns.values, window=window, min_count=2)
            # Handle initial NaNs
            vol_values[:window-1] = returns.iloc[:window-1].expanding(min_periods=2).std().values
            df['volatility_20'] = vol_values * np.sqrt(252)  # Annualized
        else:
            df['volatility_20'] = returns.rolling(
                window=window, min_periods=2
            ).std() * np.sqrt(252)

        # Fill any remaining NaNs
        df['volatility_20'] = df['volatility_20'].fillna(method='bfill').fillna(0.01)

        return df

    @staticmethod
    def _add_transformer_specific_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add features specifically useful for transformer models.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with features added
        """
        # Sequence position (normalized)
        df['sequence_position'] = np.arange(len(df)) / max(len(df) - 1, 1)

        # Log returns (more stable for transformers)
        df['log_returns'] = np.log(df['mid_price_close'] / df['mid_price_close'].shift(1))
        df['log_returns'] = df['log_returns'].fillna(0)

        # Extreme move detection
        returns = df['mid_price_close'].pct_change()
        return_std = returns.rolling(window=20, min_periods=1).std()
        df['is_extreme_move'] = (returns.abs() > 2 * return_std).astype(int)

        # Attention hints based on volatility
        vol_lookback = 20
        current_vol = df['atr_pct']
        historical_vol = current_vol.rolling(window=vol_lookback, min_periods=1).mean()
        df['attention_hint_lookback_20'] = (current_vol / historical_vol.clip(lower=1e-8)).clip(0, 3)

        # Time since last extreme move
        extreme_indices = df[df['is_extreme_move'] == 1].index
        df['time_since_last_extreme'] = 0

        for i in range(len(df)):
            if i in extreme_indices:
                df.loc[i, 'time_since_last_extreme'] = 0
            else:
                prev_extremes = extreme_indices[extreme_indices < i]
                if len(prev_extremes) > 0:
                    df.loc[i, 'time_since_last_extreme'] = i - prev_extremes[-1]
                else:
                    df.loc[i, 'time_since_last_extreme'] = i

        # Normalize time since extreme
        df['time_since_last_extreme'] = df['time_since_last_extreme'] / df['time_since_last_extreme'].max()

        # Cumulative return features
        df['cumulative_return_100'] = df['mid_price_close'] / df['mid_price_close'].shift(100) - 1
        df['cumulative_return_100'] = df['cumulative_return_100'].fillna(0)

        # Price acceleration (second derivative)
        price_change = df['mid_price_close'].pct_change()
        df['price_acceleration'] = price_change.diff()
        df['price_acceleration'] = df['price_acceleration'].fillna(0)

        return df

    @staticmethod
    def _add_attention_hint_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add features that help transformers focus attention.

        Note: Some features overlap with transformer_specific.

        Args:
            df: Input dataframe
            window: Window for calculations

        Returns:
            DataFrame with features added
        """
        # Skip if already added by transformer_specific
        if 'is_extreme_move' in df.columns:
            return df

        # Extreme moves
        returns = df['mid_price_close'].pct_change()
        return_std = returns.rolling(window=window, min_periods=1).std()
        df['is_extreme_move'] = (returns.abs() > 2 * return_std).astype(int)

        # Volatility-based attention
        current_vol = df['atr_pct'] if 'atr_pct' in df.columns else returns.rolling(window=window).std()
        historical_vol = current_vol.rolling(window=window, min_periods=1).mean()
        df['attention_hint_lookback_20'] = (current_vol / historical_vol.clip(lower=1e-8)).clip(0, 3)

        # Time since extreme
        extreme_indices = df[df['is_extreme_move'] == 1].index
        df['time_since_last_extreme'] = 0

        for i in range(len(df)):
            if i in extreme_indices:
                df.loc[i, 'time_since_last_extreme'] = 0
            else:
                prev_extremes = extreme_indices[extreme_indices < i]
                if len(prev_extremes) > 0:
                    df.loc[i, 'time_since_last_extreme'] = i - prev_extremes[-1]
                else:
                    df.loc[i, 'time_since_last_extreme'] = i

        # Normalize
        max_time = df['time_since_last_extreme'].max()
        if max_time > 0:
            df['time_since_last_extreme'] = df['time_since_last_extreme'] / max_time

        return df

    @staticmethod
    def _add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add price action features.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with features added
        """
        # Bar size relative to ATR
        bar_size = df['high_bid'] - df['low_bid']
        df['bar_size_vs_atr'] = bar_size / df['atr'].clip(lower=1e-8)

        # Wick ratios
        body_size = (df['close_bid'] - df['open_bid']).abs()
        upper_wick = df['high_bid'] - df[['close_bid', 'open_bid']].max(axis=1)
        lower_wick = df[['close_bid', 'open_bid']].min(axis=1) - df['low_bid']

        total_range = bar_size.clip(lower=1e-8)
        df['wick_ratio_upper'] = upper_wick / total_range
        df['wick_ratio_lower'] = lower_wick / total_range

        return df
