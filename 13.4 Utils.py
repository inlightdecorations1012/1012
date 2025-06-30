
"""
Utility functions for Forex Parquet Converter v13.4.
==================================================

v13.4 Changes:
- Updated get_merge_tolerance to use config multiplier
- Added get_expected_features_for_timeframe function (dynamic feature discovery)
- Updated FeatureCache to include YAML hash in cache key
- Added validation helpers for preserve mode
- Added compute_yaml_hash function
- Enhanced ForexMarketCalendar with holidays integration
- Added file locking to FeatureCache for parallel safety

Contains utilities for:
- Logging infrastructure
- Data processing helpers
- Time/calendar handling
- Memory management
- Environment setup
- Feature caching
- Feature validation
"""

import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, List, Any, Union, Set
import logging
import logging.handlers
import multiprocessing as mp
import re
import sys
import os
from pathlib import Path
from dataclasses import fields, is_dataclass
import psutil
import pickle
import json
import hashlib

# Try to import bottleneck for performance
try:
    import bottleneck as bn
    BOTTLENECK_AVAILABLE = True
except ImportError:
    bn = None
    BOTTLENECK_AVAILABLE = False

# Try to import holidays for enhanced calendar
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    holidays = None
    HOLIDAYS_AVAILABLE = False

# Try to import filelock for cache safety
try:
    import filelock
    FILELOCK_AVAILABLE = True
except ImportError:
    filelock = None
    FILELOCK_AVAILABLE = False

logger = logging.getLogger(__name__)

# Public API
__all__ = [
    # Feature functions
    'validate_feature_completeness',
    'cleanup_features_for_export',
    'get_expected_features_for_timeframe',
    # Feature cache
    'FeatureCache',
    # Logging
    'setup_logging_infrastructure',
    'get_process_logger',
    # Data processing
    'clean_forex_dataframe',
    'safe_rolling_window',
    'get_merge_tolerance',
    'collect_window_sizes',
    'parse_horizon_from_column',
    'check_feature_range',
    'fast_hash_dataframe',
    'validate_dataframe_integrity',
    'get_timeframe_rank',
    # Time/calendar
    'ForexMarketCalendar',
    # Memory management
    'estimate_memory_usage',
    'check_memory_availability',
    # Environment
    'setup_colab_environment',
    # Multiprocessing helpers
    'reconstruct_timeframe_config',
    'pickle_safe_config_dict',
    # v13.4 additions
    'compute_yaml_hash',
    'validate_timeframe_features',
    # Constants
    'BOTTLENECK_AVAILABLE'
]


# === Feature Validation Functions (Updated in v13.4) ===

def get_expected_features_for_timeframe(tf_config: 'TimeframeConfig',
                                       feature_registry: 'FeatureRegistry') -> Set[str]:
    """Get the exact features that should be present for this timeframe.

    v13.4: New function to determine expected features based on enabled_features.
           Uses dynamic discovery from FeatureRegistry instead of hard-coded map.

    Args:
        tf_config: Timeframe configuration
        feature_registry: Feature registry instance

    Returns:
        Set of expected feature names
    """
    expected_features = set()

    # Core features always present
    core_features = [
        'close_bid', 'close_ask', 'high_bid', 'high_ask',
        'low_bid', 'low_ask', 'open_bid', 'open_ask',
        'mid_price_close', 'mid_price_high', 'mid_price_low', 'mid_price_open'
    ]

    expected_features.update(core_features)

    # v13.4: Dynamic feature discovery from registry
    for feature_set in tf_config.enabled_features:
        if hasattr(feature_registry, 'get_features_for_set'):
            # Use the new method if available
            features = feature_registry.get_features_for_set(feature_set)
            expected_features.update(features)
        else:
            # Fallback to the hard-coded map (for compatibility)
            feature_map = {
                'technical_basic': [
                    'price_change', 'price_change_abs', 'ma_20', 'ma_50',
                    'price_vs_ma_20', 'price_vs_ma_50', 'rsi_14',
                    'atr', 'atr_pct', 'high_low_range', 'true_range'
                ],
                'technical_advanced': [
                    'bb_width', 'bb_position', 'macd_signal_cross'
                ],
                'microstructure_basic': [
                    'bid_ask_spread', 'spread_pct', 'volume_bid', 'volume_ask',
                    'volume_imbalance', 'has_volume_data'
                ],
                'temporal_basic': [
                    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'days_from_start'
                ],
                'session_basic': [
                    'is_london', 'is_ny', 'is_london_ny_overlap',
                    'london_open', 'ny_open', 'london_close', 'ny_close'
                ],
                'volatility_basic': [
                    'volatility_20'
                ],
                'transformer_specific': [
                    'sequence_position', 'log_returns', 'is_extreme_move',
                    'attention_hint_lookback_20', 'time_since_last_extreme',
                    'cumulative_return_100', 'price_acceleration'
                ],
                'attention_hints': [
                    'is_extreme_move', 'attention_hint_lookback_20', 'time_since_last_extreme'
                ],
                'price_action': [
                    'bar_size_vs_atr', 'wick_ratio_upper', 'wick_ratio_lower'
                ]
            }

            if feature_set in feature_map:
                expected_features.update(feature_map[feature_set])

    # Handle overlapping features (attention_hints shares with transformer_specific)
    # This is already handled by using a set

    return expected_features


def validate_timeframe_features(df: pd.DataFrame, tf_config: 'TimeframeConfig',
                               feature_registry: 'FeatureRegistry') -> Dict[str, Any]:
    """Validate features match exactly what was requested.

    v13.4: New function for per-timeframe feature validation.

    Args:
        df: DataFrame to validate
        tf_config: Timeframe configuration
        feature_registry: Feature registry instance

    Returns:
        Dictionary with validation results
    """
    expected = get_expected_features_for_timeframe(tf_config, feature_registry)

    # Get actual features (excluding targets, metadata, and normalized versions)
    actual = {
        col for col in df.columns
        if not col.startswith('target')
        and col not in ['timestamp', 'data_validity_mask']
        and not col.endswith(('_norm', '_mask', '_weight'))
        and not col.startswith('is_valid_')  # v13.4: Exclude validation masks
    }

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)

    return {
        'expected_count': len(expected),
        'actual_count': len(actual),
        'missing': missing,
        'unexpected': unexpected,
        'coverage_pct': len(expected & actual) / len(expected) * 100 if expected else 100.0,
        'complete': len(missing) == 0 and len(unexpected) == 0
    }


def validate_feature_completeness(df: pd.DataFrame, expected: List[str],
                                future_features: Optional[List[str]] = None) -> None:
    """Validate that all expected features are present and log any discrepancies.

    v13.2-rc2: Updated to handle future features separately to avoid noise.

    Args:
        df: DataFrame to validate
        expected: List of expected feature names
        future_features: Optional list of features not yet implemented
    """
    # Get actual features in the dataframe
actual_features = [
    col for col in df.columns
    if not col.startswith('target')
    and col not in ['timestamp', 'data_validity_mask']
    and not col.endswith('_norm')
    and col not in ['is_valid_price', 'is_valid_spread', 'is_valid_time_gap']  # Exclude validation columns
]

    # Find missing expected features
    missing = sorted(set(expected) - set(actual_features))

    # Find extra features not in expected or future
    all_known = set(expected)
    if future_features:
        all_known.update(future_features)

    extra = sorted(set(actual_features) - all_known)

    if missing:
        logger.warning(f"Missing features: {missing}")
    if extra:
        logger.warning(f"Unexpected features: {extra}")


def cleanup_features_for_export(df: pd.DataFrame, complete_features: List[str]) -> pd.DataFrame:
    """Keep only the specified core features plus targets and essentials.

    v13.2: Function for export cleanup.

    Args:
        df: DataFrame to clean
        complete_features: List of core features to keep

    Returns:
        DataFrame with only the specified columns
    """
    keep = (['timestamp', 'data_validity_mask'] +
        complete_features +
        [c for c in df.columns if c.startswith('target')] +
        [f'{f}_norm' for f in complete_features if f'{f}_norm' in df.columns])

# v13.4: Also keep validation masks if in preserve mode
if 'is_valid_price' in df.columns:
    keep.extend(['is_valid_price', 'is_valid_spread', 'is_valid_time_gap'])

    # Only keep columns that exist
    keep = [c for c in keep if c in df.columns]

    return df[keep]


# === Feature Cache for Incremental Enrichment (Updated in v13.4) ===

class FeatureCache:
    """Cache for incremental feature enrichment pathway.

    v13.4: Updated to include YAML hash in cache keys.

    Stores computed features to disk for reuse across runs,
    significantly speeding up repeated processing of the same data.
    """

    def __init__(self, cache_dir: Path, max_cache_size_gb: float = 10.0,
                 yaml_hash: Optional[str] = None):
        """Initialize feature cache.

        Args:
            cache_dir: Directory for cache storage
            max_cache_size_gb: Maximum cache size in GB
            yaml_hash: Optional YAML configuration hash
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_bytes = max_cache_size_gb * 1024 * 1024 * 1024
        self.yaml_hash = yaml_hash  # v13.4: Store YAML hash

        # Cache statistics
        self._hits = 0
        self._misses = 0

        # Load cache index
        self.index_file = self.cache_dir / 'cache_index.json'
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                # v13.4: Use file locking if available
                if FILELOCK_AVAILABLE:
                    lock_file = self.index_file.with_suffix('.json.lock')
                    lock = filelock.FileLock(lock_file, timeout=10)

                    with lock:
                        with open(self.index_file, 'r') as f:
                            self.index = json.load(f)
                else:
                    with open(self.index_file, 'r') as f:
                        self.index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self.index = {}
        else:
            self.index = {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            # v13.4: Use file locking if available
            if FILELOCK_AVAILABLE:
                lock_file = self.index_file.with_suffix('.json.lock')
                lock = filelock.FileLock(lock_file, timeout=10)

                with lock:
                    with open(self.index_file, 'w') as f:
                        json.dump(self.index, f, indent=2)
            else:
                with open(self.index_file, 'w') as f:
                    json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")

    def get(self, key: str) -> Optional[Dict[str, pd.Series]]:
        """Get features from cache.

        Args:
            key: Cache key

        Returns:
            Dictionary of features or None if not found
        """
        # v13.4: Include YAML hash in the actual cache key
        if self.yaml_hash:
            full_key = f"{self.yaml_hash}_{key}"
        else:
            full_key = key

        if full_key not in self.index:
            self._misses += 1
            return None

        cache_file = self.cache_dir / f"{full_key}.pkl"
        if not cache_file.exists():
            self._misses += 1
            return None

        try:
            # v13.4: Use file locking if available for parallel safety
            if FILELOCK_AVAILABLE:
                lock_file = cache_file.with_suffix('.pkl.lock')
                lock = filelock.FileLock(lock_file, timeout=10)

                with lock:
                    with open(cache_file, 'rb') as f:
                        features = pickle.load(f)
            else:
                # Fallback to direct read
                with open(cache_file, 'rb') as f:
                    features = pickle.load(f)

            self._hits += 1
            return features
        except Exception as e:
            logger.warning(f"Failed to load cached features: {e}")
            self._misses += 1
            return None

    def set(self, key: str, features: Dict[str, pd.Series]) -> None:
        """Store features in cache.

        Args:
            key: Cache key
            features: Dictionary of features to cache
        """
        # v13.4: Include YAML hash in the actual cache key
        if self.yaml_hash:
            full_key = f"{self.yaml_hash}_{key}"
        else:
            full_key = key

        # Check cache size
        self._enforce_size_limit()

        cache_file = self.cache_dir / f"{full_key}.pkl"

        try:
            # v13.4: Use file locking if available for parallel safety
            if FILELOCK_AVAILABLE:
                lock_file = cache_file.with_suffix('.pkl.lock')
                lock = filelock.FileLock(lock_file, timeout=10)

                with lock:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                # Fallback to direct write (not parallel-safe)
                with open(cache_file, 'wb') as f:
                    pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Update index
            self.index[full_key] = {
                'timestamp': datetime.now().isoformat(),
                'size_bytes': cache_file.stat().st_size
            }
            self._save_index()

        except Exception as e:
            logger.error(f"Failed to cache features: {e}")

    def _enforce_size_limit(self) -> None:
        """Remove old entries if cache exceeds size limit."""
        total_size = sum(
            entry.get('size_bytes', 0)
            for entry in self.index.values()
        )

        if total_size > self.max_cache_size_bytes:
            # Sort by timestamp and remove oldest
            sorted_entries = sorted(
                self.index.items(),
                key=lambda x: x[1].get('timestamp', ''),
                reverse=False
            )

            while total_size > self.max_cache_size_bytes * 0.9:  # Keep 10% buffer
                if not sorted_entries:
                    break

                key_to_remove, entry = sorted_entries.pop(0)
                cache_file = self.cache_dir / f"{key_to_remove}.pkl"

                if cache_file.exists():
                    cache_file.unlink()

                total_size -= entry.get('size_bytes', 0)
                del self.index[key_to_remove]

            self._save_index()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary of statistics
        """
        total_size = sum(
            (self.cache_dir / f"{key}.pkl").stat().st_size
            for key in self.index
            if (self.cache_dir / f"{key}.pkl").exists()
        )

        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / max(1, self._hits + self._misses),
            'num_entries': len(self.index),
            'total_size_mb': total_size / (1024 * 1024),
            'max_size_mb': self.max_cache_size_bytes / (1024 * 1024),
            'yaml_hash': self.yaml_hash[:8] if self.yaml_hash else 'None'  # v13.4
        }

    def get_hit_rate(self) -> float:
        """Get cache hit rate.

        Returns:
            Hit rate between 0 and 1
        """
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def cleanup(self) -> None:
        """Clean up cache and save index."""
        self._save_index()
        logger.info(f"Feature cache stats: {self.get_stats()}")


# === v13.4 Utility Functions ===

def compute_yaml_hash(config_path: Path) -> str:
    """Compute SHA-256 hash of YAML configuration file.

    v13.4: New function for configuration tracking.

    Args:
        config_path: Path to YAML file

    Returns:
        Hex string of first 16 characters of hash
    """
    if not config_path.exists():
        return "no_config"

    try:
        with open(config_path, 'rb') as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()[:16]
    except Exception as e:
        logger.warning(f"Could not hash YAML config: {e}")
        return "hash_error"


# === Logging Infrastructure ===

def setup_logging_infrastructure(log_queue: mp.Queue, log_level: str = 'INFO',
                               non_blocking: bool = False) -> None:
    """Setup logging infrastructure for the current process using queue handler.

    Fixed to support non-blocking queue operations with fallback.

    Args:
        log_queue: Multiprocessing queue for log records
        log_level: Logging level as string
        non_blocking: If True, use non-blocking put with fallback
    """
    # Get root logger
    logger = logging.getLogger()

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create queue handler
    queue_handler = logging.handlers.QueueHandler(log_queue)

    # Override enqueue for non-blocking operation
    if non_blocking:
        # Store original enqueue method
        original_enqueue = queue_handler.enqueue

        # Create fallback console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

        def non_blocking_enqueue(record):
            """Non-blocking enqueue with fallback to console."""
            try:
                # Try non-blocking put
                log_queue.put(record, block=False)
            except Exception:
                # Queue is full or dead, fallback to console
                console_handler.handle(record)

        # Replace enqueue method
        queue_handler.enqueue = non_blocking_enqueue

    logger.addHandler(queue_handler)

    # Set level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Also configure the module loggers
    for name in ['__main__', 'converter', 'config', 'pipeline', 'utils']:
        module_logger = logging.getLogger(name)
        module_logger.setLevel(level)


def get_process_logger(name: str) -> logging.Logger:
    """Get a logger configured for the current process.

    Args:
        name: Logger name

    Returns:
        Configured logger
    """
    return logging.getLogger(name)


# === Data Processing Helpers ===

def clean_forex_dataframe(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Očisti raw forex-CSV (BID ili ASK) i presloži nazive u open_bid / open_ask …
    CSV očekuje zaglavlje: 'Gmt time,Open,High,Low,Close,Volume'
    Args:
        df     – sirovi DataFrame iz pd.read_csv
        suffix – 'bid' ili 'ask' (dolazi iz Convertera)
    Returns:
        očišćeni DataFrame sa stupcima:
        timestamp, open_<suffix>, high_<suffix>, low_<suffix>, close_<suffix>, volume_<suffix>
    """

    # 1) normaliziraj nazive (trim + lower)
    df.columns = [c.strip().lower() for c in df.columns]

    # 2) mapiraj 'gmt time' → 'timestamp'
    df = df.rename(columns={'gmt time': 'timestamp', 'gmt_time': 'timestamp',
                            'time': 'timestamp'})

    if 'timestamp' not in df.columns:
        logger.error("CSV nema stupac s vremenom – preskačem chunk")
        return pd.DataFrame()

    # 3) parsiraj datum (dvije varijante formata)
    original_ts = df['timestamp'].copy()
    df['timestamp'] = pd.to_datetime(original_ts,
                                     format='%d.%m.%Y %H:%M:%S.%f',
                                     errors='coerce',
                                     utc=True)
    if df['timestamp'].isna().all():
        # fallback na ISO 'YYYY-MM-DD HH:MM:SS'
        df['timestamp'] = pd.to_datetime(original_ts, errors='coerce', utc=True)

    df = df.dropna(subset=['timestamp'])
    if df.empty:
        return df

    # 4) preimenuj cijene + volumen
    col_map = {
        'open':  f'open_{suffix}',
        'high':  f'high_{suffix}',
        'low':   f'low_{suffix}',
        'close': f'close_{suffix}',
        'volume': f'volume_{suffix}',
    }
    df = df.rename(columns=col_map)

    # 5) zadrži samo relevantne stupce
    keep = ['timestamp'] + list(col_map.values())
    df = df[[c for c in keep if c in df.columns]]

    # 6) u numeričkim stupcima konvertiraj na float i ukloni nevaljane retke
    price_cols = [c for c in keep if c.startswith(('open_', 'high_', 'low_', 'close_'))]
    for c in price_cols + [f'volume_{suffix}']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=price_cols)
    for c in price_cols:
        df = df[df[c] > 0]

    # Volumen NaN → 0
    vol_col = f'volume_{suffix}'
    if vol_col in df.columns:
        df[vol_col] = df[vol_col].fillna(0)

    return df.reset_index(drop=True)


def safe_rolling_window(series: pd.Series, window: int, func: str,
                       min_periods: Optional[int] = None) -> pd.Series:
    """Apply rolling window function with proper edge case handling.

    Args:
        series: Input series
        window: Window size
        func: Function name ('mean', 'std', etc.)
        min_periods: Minimum periods for calculation

    Returns:
        Series with rolling calculation applied
    """
    if len(series) == 0:
        return series

    if min_periods is None:
        min_periods = 1

    # For very small windows, use expanding
    if window > len(series):
        expanding_result = getattr(series.expanding(min_periods=min_periods), func)()
        return expanding_result

    # Normal rolling window
    result = getattr(series.rolling(window=window, min_periods=min_periods), func)()

    # Handle initial NaN values with expanding window
    if result.isna().any():
        expanding_fill = getattr(series.expanding(min_periods=min_periods), func)()
        result = result.fillna(expanding_fill)

    # Special handling for std to prevent division by zero
    if func == 'std':
        # First fill NaNs, then clip to minimum value
        if result.isna().any():
            series_std = series.std()
            if pd.isna(series_std) or series_std == 0:
                series_std = 1e-8
            result = result.fillna(series_std)
        # Then clip to prevent zero values
        result = result.clip(lower=1e-8)
    elif func == 'mean':
        # For mean, use expanding mean to avoid look-ahead bias
        if result.isna().any():
            expanding_mean = series.expanding(min_periods=min_periods).mean()
            result = result.fillna(expanding_mean)
    else:
        # For other functions, use series aggregate as fallback
        if result.isna().any():
            fallback_value = getattr(series, func)()
            if pd.isna(fallback_value):
                fallback_value = 0 if func == 'mean' else series.iloc[0]
            result = result.fillna(fallback_value)

    return result


def get_merge_tolerance(timeframe: str, multiplier: float = 0.5,
                       config: Optional['ProcessingConfig'] = None) -> pd.Timedelta:
    """Get appropriate merge tolerance for timeframe.

    v13.4: Updated to use config multiplier if provided.

    Args:
        timeframe: Timeframe string (e.g., '15T', '1H')
        multiplier: Fraction of bar duration for tolerance (deprecated)
        config: Optional processing config with merge_tolerance_multiplier

    Returns:
        Timedelta for merge tolerance
    """
    # Use config multiplier if available
    if config and hasattr(config, 'merge_tolerance_multiplier'):
        multiplier = config.merge_tolerance_multiplier

    # Base tolerances (full bar duration)
    base_tolerances = {
        '15T': pd.Timedelta('15min'),
        '15min': pd.Timedelta('15min'),
        '1H': pd.Timedelta('1h'),
        '1h': pd.Timedelta('1h'),
        '4H': pd.Timedelta('4h'),
        '4h': pd.Timedelta('4h'),
        '1D': pd.Timedelta('1d'),
        '1d': pd.Timedelta('1d'),
        '1W': pd.Timedelta('7d'),
        '1w': pd.Timedelta('7d')
    }

    base = base_tolerances.get(timeframe, pd.Timedelta('1min'))
    return base * multiplier


def collect_window_sizes(params: dict, path: str = "") -> List[int]:
    """Extract all window/period parameters from nested config.

    Args:
        params: Parameter dictionary
        path: Current path in nested structure

    Returns:
        List of window sizes found
    """
    windows = []

    for key, value in params.items():
        current_path = f"{path}.{key}" if path else key

        if isinstance(value, dict):
            # Recurse into nested dictionaries
            windows.extend(collect_window_sizes(value, current_path))
        elif any(term in key.lower() for term in ['period', 'window', 'span']):
            # Found a window parameter
            if isinstance(value, list):
                windows.extend(value)
            elif isinstance(value, (int, float)):
                windows.append(int(value))

    return windows


def parse_horizon_from_column(col: str) -> Optional[int]:
    """Parse horizon value from target column name.

    Args:
        col: Column name like 'target_h12_soft'

    Returns:
        Horizon value or None if not found
    """
    # Use regex to extract horizon value
    match = re.search(r'_h(\d+)', col)
    if match:
        return int(match.group(1))
    return None


def check_feature_range(series: pd.Series, pattern: str,
                       expected_range: Tuple[float, float]) -> bool:
    """Check if feature values are within expected range.

    Args:
        series: Feature series to check
        pattern: Pattern to match (e.g., '_sin', 'rsi_')
        expected_range: (min, max) tuple

    Returns:
        True if all values in range
    """
    # Use word boundaries for exact matching
    if pattern.startswith('_'):
        # Suffix pattern
        regex = rf'\w+{re.escape(pattern)}$'
    elif pattern.endswith('_'):
        # Prefix pattern
        regex = rf'^{re.escape(pattern)}\w+'
    else:
        # Anywhere in the name
        regex = re.escape(pattern)

    # Check if column name matches pattern
    if not re.search(regex, series.name):
        return True  # Not applicable

    # Check range
    min_val, max_val = expected_range
    actual_min = series.min()
    actual_max = series.max()

    return actual_min >= min_val and actual_max <= max_val


def fast_hash_dataframe(df: pd.DataFrame, sample_size: int = 10000) -> str:
    """Compute fast hash of dataframe for checksums.

    Includes more columns and metadata for better uniqueness.

    Args:
        df: DataFrame to hash
        sample_size: Number of rows to sample

    Returns:
        Hex hash string
    """
    # Sample rows if dataframe is large
    if len(df) > sample_size:
        # Use deterministic sampling
        step = len(df) // sample_size
        sample_df = df.iloc[::step].head(sample_size)
    else:
        sample_df = df

    # Include shape and column names in hash
    shape_str = f"{df.shape[0]}x{df.shape[1]}"
    cols_str = ','.join(sorted(df.columns))

    # Hash the sampled data
    try:
        # Use pandas utility for object hashing
        data_hash = pd.util.hash_pandas_object(sample_df, index=False)
        combined_hash = hashlib.md5(
            f"{shape_str}_{cols_str}_{data_hash.sum()}".encode()
        ).hexdigest()
    except Exception:
        # Fallback to simpler hash
        combined_hash = hashlib.md5(
            f"{shape_str}_{cols_str}_{sample_df.values.tobytes()}".encode()
        ).hexdigest()

    return combined_hash


def validate_dataframe_integrity(df: pd.DataFrame, stage_name: str) -> None:
    """Validate dataframe integrity at pipeline stages.

    Args:
        df: DataFrame to validate
        stage_name: Name of pipeline stage

    Raises:
        ValueError: If integrity checks fail
    """
    # Check for empty dataframe
    if df.empty:
        raise ValueError(f"{stage_name}: DataFrame is empty")

    # Check for duplicate timestamps
    if 'timestamp' in df.columns:
        dup_count = df['timestamp'].duplicated().sum()
        if dup_count > 0:
            raise ValueError(f"{stage_name}: Found {dup_count} duplicate timestamps")

    # Check for all-NaN columns
    all_nan_cols = df.columns[df.isna().all()].tolist()
    if all_nan_cols:
        logger.warning(f"{stage_name}: Columns with all NaN values: {all_nan_cols}")

    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    inf_cols = inf_counts[inf_counts > 0].index.tolist()
    if inf_cols:
        raise ValueError(f"{stage_name}: Found infinite values in columns: {inf_cols}")

    # Check data types consistency
    expected_dtypes = {
        'timestamp': 'datetime64[ns, UTC]',
        'volume_bid': 'float',
        'volume_ask': 'float'
    }

    for col, expected_dtype in expected_dtypes.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            if 'datetime' in expected_dtype and 'datetime' not in actual_dtype:
                logger.warning(f"{stage_name}: Column {col} has dtype {actual_dtype}, expected {expected_dtype}")


def get_timeframe_rank(timeframe: str) -> int:
    """Get numeric rank for timeframe for sorting.

    Args:
        timeframe: Timeframe string

    Returns:
        Numeric rank (lower is higher frequency)
    """
    rank_map = {
        '15T': 1, '15min': 1,
        '1H': 2, '1h': 2,
        '4H': 3, '4h': 3,
        '1D': 4, '1d': 4,
        '1W': 5, '1w': 5
    }
    return rank_map.get(timeframe, 99)


# === Time/Calendar Functions (Updated in v13.4) ===

class ForexMarketCalendar:
    """Forex market calendar for holiday detection.

    v13.4: Enhanced with holidays package integration.
    """

    def __init__(self, calendar_mode: str = 'basic'):
        """Initialize calendar.

        Args:
            calendar_mode: 'basic' or 'full' for holiday coverage
        """
        self.calendar_mode = calendar_mode

        # Basic holidays (always included)
        self.basic_holidays = {
            'new_year': (1, 1),
            'christmas': (12, 25),
            'christmas_eve': (12, 24)
        }

        # Initialize country calendars if available and in full mode
        self.us_holidays = None
        self.uk_holidays = None

        if calendar_mode == 'full' and HOLIDAYS_AVAILABLE:
            try:
                # Initialize for current year and next few years
                current_year = datetime.now().year
                self.us_holidays = holidays.US(years=range(current_year-2, current_year+3))
                self.uk_holidays = holidays.UK(years=range(current_year-2, current_year+3))
                logger.info("Initialized full holiday calendar with US and UK holidays")
            except Exception as e:
                logger.warning(f"Could not initialize full holiday calendar: {e}")

    def is_forex_closed(self, dt: datetime) -> bool:
        """Check if forex market is closed at given time.

        Args:
            dt: Datetime to check

        Returns:
            True if market is closed
        """
        # Weekend check (Friday 22:00 UTC to Sunday 22:00 UTC)
        if dt.weekday() == 5:  # Saturday
            return True
        elif dt.weekday() == 6:  # Sunday
            if dt.hour < 22:
                return True
        elif dt.weekday() == 4:  # Friday
            if dt.hour >= 22:
                return True

        # Holiday check
        date_only = dt.date()

        # Basic holidays
        for holiday_name, (month, day) in self.basic_holidays.items():
            if date_only.month == month and date_only.day == day:
                return True

        # v13.4: Full holiday check if enabled
        if self.calendar_mode == 'full' and HOLIDAYS_AVAILABLE:
            # Check US holidays
            if self.us_holidays and date_only in self.us_holidays:
                return True

            # Check UK holidays
            if self.uk_holidays and date_only in self.uk_holidays:
                return True

        return False

    def next_open_time(self, dt: datetime) -> datetime:
        """Get next market open time after given datetime.

        Args:
            dt: Current datetime

        Returns:
            Next market open datetime
        """
        next_dt = dt

        while self.is_forex_closed(next_dt):
            next_dt += pd.Timedelta(hours=1)

        return next_dt


# === Memory Management Functions ===

def estimate_memory_usage(df: pd.DataFrame, multiplier: float = 2.5) -> float:
    """Estimate memory usage for processing.

    Args:
        df: DataFrame to estimate
        multiplier: Safety multiplier for processing overhead

    Returns:
        Estimated memory usage in GB
    """
    # Get current memory usage
    current_usage = df.memory_usage(deep=True).sum() / (1024 ** 3)

    # Estimate with features and intermediate calculations
    estimated_usage = current_usage * multiplier

    return estimated_usage


def check_memory_availability(required_gb: Optional[float] = None) -> float:
    """Check available system memory.

    Args:
        required_gb: Optional required memory in GB

    Returns:
        Available memory in GB

    Raises:
        MemoryError: If insufficient memory
    """
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)

    logger.info(f"System memory: {mem.total / (1024 ** 3):.1f} GB total, "
               f"{available_gb:.1f} GB available ({mem.percent:.1f}% used)")

    if required_gb and available_gb < required_gb:
        raise MemoryError(
            f"Insufficient memory: {available_gb:.1f} GB available, "
            f"{required_gb:.1f} GB required"
        )

    return available_gb


# === Environment Setup ===

def setup_colab_environment() -> bool:
    """Setup Google Colab environment if detected.

    Returns:
        True if in Colab environment
    """
    try:
        import google.colab
        in_colab = True

        # Mount Google Drive if not already mounted
        try:
            from google.colab import drive
            if not os.path.exists('/content/drive'):
                logger.info("Mounting Google Drive...")
                drive.mount('/content/drive')
        except ImportError:
            logger.warning("google.colab.drive could not be imported. Skipping Google Drive mount.")

        # Set environment variables for better performance
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

        logger.info("Google Colab environment detected and configured")

    except ImportError:
        in_colab = False
        logger.info("Not running in Google Colab")

    return in_colab


# === Multiprocessing Helpers ===

def reconstruct_timeframe_config(config_dict: Dict[str, Any]) -> 'TimeframeConfig':
    """Reconstruct TimeframeConfig from dictionary representation.

    This is needed because asdict() flattens nested dataclasses to dicts.

    Args:
        config_dict: Dictionary representation of TimeframeConfig

    Returns:
        Properly reconstructed TimeframeConfig instance
    """
    from config import (
        TimeframeConfig, TargetConfig, NormalizationConfig,
        QualityControlConfig, DebugConfig
    )

    # Reconstruct nested dataclasses
    if 'target_config' in config_dict:
        config_dict['target_config'] = TargetConfig(**config_dict['target_config'])

    if 'normalization_config' in config_dict:
        config_dict['normalization_config'] = NormalizationConfig(**config_dict['normalization_config'])

    if 'quality_control_config' in config_dict:
        config_dict['quality_control_config'] = QualityControlConfig(**config_dict['quality_control_config'])

    if 'debug_config' in config_dict:
        config_dict['debug_config'] = DebugConfig(**config_dict['debug_config'])

    return TimeframeConfig(**config_dict)


def pickle_safe_config_dict(tf_config: 'TimeframeConfig') -> Dict[str, Any]:
    """Convert TimeframeConfig to a pickle-safe dictionary.

    Args:
        tf_config: TimeframeConfig instance

    Returns:
        Dictionary safe for pickling
    """
    from dataclasses import asdict
    return asdict(tf_config)
