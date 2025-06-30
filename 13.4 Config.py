
"""
Configuration classes for Forex Parquet Converter v13.4.
========================================================

v13.4 Changes:
- Added data_version field for dynamic version handling
- Added checksum_mode and calendar to ProcessingConfig
- Added configurable thresholds to QualityControlConfig
- Updated validation_mode field in QualityControlConfig

Contains dataclasses for:
- Path configuration
- Target configuration
- Normalization configuration
- Quality control configuration
- Processing configuration
- Feature configuration
- Timeframe configuration
- Main ForexConfig container
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Custom exceptions
class ForexDataError(Exception):
    """Base exception for forex data errors."""
    pass

class ConfigurationError(ForexDataError):
    """Configuration-related errors."""
    pass

class DataQualityError(ForexDataError):
    """Data quality validation errors."""
    pass


@dataclass
class PathConfig:
    """Path configuration."""
    input: str
    output: str

    def __post_init__(self) -> None:
        """Validate input path exists and convert to Path objects."""
        self.input = Path(self.input)
        self.output = Path(self.output)

        if not self.input.exists():
            raise ConfigurationError(f"Input path does not exist: {self.input}")


@dataclass
class TargetConfig:
    """Target generation configuration."""
    type: str = 'multi_horizon'
    lookahead_periods: int = 12
    atr_multiplier: float = 0.7
    multi_horizon: List[int] = field(default_factory=lambda: [1, 3, 6, 12])
    use_soft_labels: bool = True
    # v13.4: New fields for dynamic ATR
    dynamic_atr_multiplier: bool = True
    target_minority_class_pct: float = 0.03

    def __post_init__(self) -> None:
        """Validate target configuration."""
        if self.type not in ['single_horizon', 'multi_horizon']:
            raise ConfigurationError(f"Unknown target type: {self.type}")
        if self.atr_multiplier <= 0:
            raise ConfigurationError("atr_multiplier must be positive")
        if self.lookahead_periods < 1:
            raise ConfigurationError("lookahead_periods must be >= 1")
        if self.target_minority_class_pct <= 0 or self.target_minority_class_pct >= 0.5:
            raise ConfigurationError("target_minority_class_pct must be between 0 and 0.5")


@dataclass
class NormalizationConfig:
    """Normalization configuration."""
    method: str = 'zscore'
    window: int = 252 * 4
    min_periods: int = 50

    def __post_init__(self) -> None:
        """Validate normalization configuration."""
        if self.window < self.min_periods:
            raise ConfigurationError("window must be >= min_periods")
        if self.min_periods < 1:
            raise ConfigurationError("min_periods must be positive")

        # v13.1: Accept both 'zscore' and 'robust_zscore' for backward compatibility
        # Map 'robust_zscore' to 'zscore' since implementation is standard z-score
        if self.method == 'robust_zscore':
            self.method = 'zscore'

        if self.method not in ['zscore', 'minmax', 'standard']:
            raise ConfigurationError(
                f"Unknown normalization method: {self.method}. "
                f"Valid methods: zscore, minmax, standard"
            )


@dataclass
class QualityControlConfig:
    """Quality control thresholds configuration."""
    min_rows_per_timeframe: int = 1000
    max_spread_pct: float = 0.01
    min_spread_pct: float = 0.00001
    max_time_gap_multiplier: float = 5.0
    max_class_imbalance_ratio: float = 20.0
    # v13.4: New field for validation mode
    validation_mode: str = 'drop'  # 'drop' or 'preserve'
    # v13.4: Configurable validation thresholds
    spread_multiplier: float = 4.0
    spread_floor: float = 0.002
    atr_grid_start: float = 0.1
    atr_grid_stop: float = 5.0
    atr_grid_step: float = 0.1

    def __post_init__(self) -> None:
        """Validate quality control configuration."""
        if self.min_spread_pct >= self.max_spread_pct:
            raise ConfigurationError("min_spread_pct must be < max_spread_pct")
        if self.max_time_gap_multiplier <= 0:
            raise ConfigurationError("max_time_gap_multiplier must be positive")
        if self.max_class_imbalance_ratio <= 1:
            raise ConfigurationError("max_class_imbalance_ratio must be > 1")
        if self.min_rows_per_timeframe < 1:
            raise ConfigurationError("min_rows_per_timeframe must be positive")
        if self.validation_mode not in ['drop', 'preserve']:
            raise ConfigurationError("validation_mode must be 'drop' or 'preserve'")
        if self.spread_multiplier <= 0:
            raise ConfigurationError("spread_multiplier must be positive")
        if self.spread_floor <= 0:
            raise ConfigurationError("spread_floor must be positive")
        if self.atr_grid_start <= 0 or self.atr_grid_stop <= 0 or self.atr_grid_step <= 0:
            raise ConfigurationError("ATR grid parameters must be positive")
        if self.atr_grid_start >= self.atr_grid_stop:
            raise ConfigurationError("atr_grid_start must be < atr_grid_stop")


@dataclass
class DebugConfig:
    """Debug options configuration."""
    save_intermediate_csvs: bool = False
    limit_rows: Optional[int] = None
    verbose_logging: bool = False

    def __post_init__(self) -> None:
        """Validate debug configuration."""
        if self.limit_rows is not None and self.limit_rows <= 0:
            raise ConfigurationError("limit_rows must be positive or None")


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    parallel_enabled: bool = True
    max_workers: int = 2
    chunk_size: int = 50000
    stream_read: bool = True
    partitioning_enabled: bool = True
    # v13.4: New field for merge tolerance
    merge_tolerance_multiplier: float = 0.1  # Reduced from 0.5
    # v13.4: New fields for checksum and calendar
    checksum_mode: str = 'metadata'  # 'metadata' or 'content'
    calendar: str = 'basic'  # 'basic' or 'full'

    def __post_init__(self) -> None:
        """Validate processing configuration."""
        if self.max_workers < 1:
            raise ConfigurationError("max_workers must be >= 1")
        if self.chunk_size < 1000:
            raise ConfigurationError("chunk_size must be >= 1000 for efficiency")
        if self.merge_tolerance_multiplier <= 0 or self.merge_tolerance_multiplier > 1:
            raise ConfigurationError("merge_tolerance_multiplier must be between 0 and 1")
        if self.checksum_mode not in ['metadata', 'content']:
            raise ConfigurationError("checksum_mode must be 'metadata' or 'content'")
        if self.calendar not in ['basic', 'full']:
            raise ConfigurationError("calendar must be 'basic' or 'full'")


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    save_to_file: bool = True

    def __post_init__(self) -> None:
        """Validate logging configuration."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.level.upper() not in valid_levels:
            raise ConfigurationError(
                f"Invalid log level: {self.level}. "
                f"Must be one of: {', '.join(valid_levels)}"
            )


@dataclass
class TimeframeConfig:
    """Configuration for a single timeframe."""
    name: str
    file_pattern: str
    frequency: str
    timezone: str = 'UTC'
    chunk_size: int = 50000
    enabled_features: List[str] = field(default_factory=list)
    feature_params: Dict[str, Any] = field(default_factory=dict)
    target_config: TargetConfig = field(default_factory=TargetConfig)
    normalization_config: NormalizationConfig = field(default_factory=NormalizationConfig)
    quality_control_config: QualityControlConfig = field(default_factory=QualityControlConfig)
    debug_config: DebugConfig = field(default_factory=DebugConfig)

    def __post_init__(self) -> None:
        """Validate and normalize timeframe configuration."""
        # Validate name
        if not self.name:
            raise ConfigurationError("Timeframe name cannot be empty")

        # Validate and standardize frequency
        valid_frequencies = ['15T', '15min', '1H', '1h', '4H', '4h', '1D', '1d', '1W', '1w']
        if self.frequency not in valid_frequencies:
            # Try to standardize
            freq_map = {
                '15M': '15T', '15MIN': '15T', '15m': '15T', '15Min': '15T',
                '1HOUR': '1H', '1hour': '1H', '1Hr': '1H',
                '4HOUR': '4H', '4hour': '4H', '4Hr': '4H',
                '1DAY': '1D', '1day': '1D', '1Day': '1D',
                '1WEEK': '1W', '1week': '1W', '1Week': '1W'
            }
            self.frequency = freq_map.get(self.frequency, self.frequency)

            # If still not valid, try pandas frequency parsing
            if self.frequency not in valid_frequencies:
                try:
                    # Attempt to parse with pandas to validate
                    pd.tseries.frequencies.to_offset(self.frequency)
                    # If successful, keep the frequency as-is
                    logger.info(f"Using non-standard but valid frequency: {self.frequency}")
                except Exception:
                    # If pandas can't parse it either, raise error
                    raise ConfigurationError(
                        f"Invalid frequency: {self.frequency}. "
                        f"Must be one of: {', '.join(valid_frequencies)} or a valid pandas frequency string"
                    )

        # Validate file pattern
        if not self.file_pattern:
            raise ConfigurationError("file_pattern cannot be empty")

        # Validate chunk size
        if self.chunk_size < 1000:
            raise ConfigurationError("chunk_size must be >= 1000")

        # Validate enabled features
        valid_features = [
            'technical_basic', 'technical_advanced', 'microstructure_basic',
            'temporal_basic', 'session_basic', 'volatility_basic',
            'transformer_specific', 'attention_hints', 'price_action'
        ]
        for feature in self.enabled_features:
            if feature not in valid_features:
                raise ConfigurationError(
                    f"Unknown feature: {feature}. "
                    f"Valid features: {', '.join(valid_features)}"
                )


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""
    technical_basic: Dict[str, Any] = field(default_factory=lambda: {
        'ma_periods': [20, 50],
        'rsi_period': 14
    })
    technical_advanced: Dict[str, Any] = field(default_factory=lambda: {
        'bb_period': 20,
        'bb_std': 2
    })
    volatility_basic: Dict[str, Any] = field(default_factory=lambda: {
        'window': 20,
        'atr_period': 14
    })
    attention_hints: Dict[str, Any] = field(default_factory=lambda: {
        'window': 20
    })

    def __post_init__(self) -> None:
        """Validate feature parameters."""
        # Validate MA periods
        if 'ma_periods' in self.technical_basic:
            periods = self.technical_basic['ma_periods']
            if not isinstance(periods, list) or not periods:
                raise ConfigurationError("ma_periods must be a non-empty list")
            if any(p <= 0 for p in periods):
                raise ConfigurationError("All MA periods must be positive")

        # Validate other numeric parameters
        numeric_params = [
            ('technical_basic', 'rsi_period'),
            ('technical_advanced', 'bb_period'),
            ('technical_advanced', 'bb_std'),
            ('volatility_basic', 'window'),
            ('volatility_basic', 'atr_period'),
            ('attention_hints', 'window')
        ]

        for config_name, param_name in numeric_params:
            config_dict = getattr(self, config_name, {})
            if param_name in config_dict:
                value = config_dict[param_name]
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ConfigurationError(
                        f"{config_name}.{param_name} must be a positive number"
                    )


@dataclass
class ForexConfig:
    """Main configuration container."""
    # v13.4: Added data_version field
    data_version: str = "13.4"
    paths: PathConfig = field(default_factory=PathConfig)
    timeframes: List[TimeframeConfig] = field(default_factory=list)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    quality_control: QualityControlConfig = field(default_factory=QualityControlConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def __post_init__(self) -> None:
        """Validate main configuration."""
        if not self.timeframes:
            raise ConfigurationError("At least one timeframe must be configured")

        # Check for duplicate timeframe names
        names = [tf.name for tf in self.timeframes]
        if len(names) != len(set(names)):
            raise ConfigurationError("Duplicate timeframe names found")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ForexConfig':
        """Create ForexConfig from dictionary.

        Args:
            config_dict: Configuration dictionary from YAML

        Returns:
            ForexConfig instance

        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # v13.4: Extract data_version if present
            data_version = config_dict.get('data_version', '13.4')

            # Parse paths
            paths = PathConfig(**config_dict['paths'])

            # Parse timeframes with deep copies to prevent sharing
            timeframes = []
            qc_base = config_dict.get('quality_control', {})
            debug_base = config_dict.get('debug', {})

            for tf_dict in config_dict.get('timeframes', []):
                # Create target config
                target_cfg = tf_dict.get('target', {})
                target = TargetConfig(
                    type=target_cfg.get('type', 'multi_horizon'),
                    lookahead_periods=target_cfg.get('lookahead_periods', 12),
                    atr_multiplier=target_cfg.get('atr_multiplier', 0.7),
                    multi_horizon=target_cfg.get('multi_horizon', [1, 3, 6, 12]),
                    use_soft_labels=target_cfg.get('use_soft_labels', True),
                    # v13.4: New fields
                    dynamic_atr_multiplier=target_cfg.get('dynamic_atr_multiplier', True),
                    target_minority_class_pct=target_cfg.get('target_minority_class_pct', 0.03)
                )

                # Create normalization config
                norm_cfg = tf_dict.get('normalization', {})
                normalization = NormalizationConfig(**norm_cfg)

                # Create quality control config with overrides
                qc_dict = {**qc_base, **tf_dict.get('quality_control', {})}
                quality_control = QualityControlConfig(**qc_dict)

                # Create debug config with overrides
                debug_dict = {**debug_base, **tf_dict.get('debug', {})}
                debug_control = DebugConfig(**debug_dict)

                # Get feature params from the global features section
                feature_params = config_dict.get('features', {})

                # Create timeframe config
                tf_config = TimeframeConfig(
                    name=tf_dict['name'],
                    file_pattern=tf_dict['file_pattern'],
                    frequency=tf_dict['frequency'],
                    timezone=tf_dict.get('timezone', 'UTC'),
                    chunk_size=tf_dict.get('chunk_size', 50000),
                    enabled_features=tf_dict.get('enabled_features', []),
                    feature_params=feature_params,
                    target_config=target,
                    normalization_config=normalization,
                    quality_control_config=quality_control,
                    debug_config=debug_control
                )

                timeframes.append(tf_config)

            # Parse other configs
            features = FeatureConfig(**config_dict.get('features', {}))
            processing = ProcessingConfig(**config_dict.get('processing', {}))
            logging = LoggingConfig(**config_dict.get('logging', {}))
            quality_control = QualityControlConfig(**config_dict.get('quality_control', {}))
            debug = DebugConfig(**config_dict.get('debug', {}))

            return cls(
                data_version=data_version,
                paths=paths,
                timeframes=timeframes,
                features=features,
                processing=processing,
                logging=logging,
                quality_control=quality_control,
                debug=debug
            )

        except TypeError as e:
            raise ConfigurationError(f"Invalid configuration structure: {e}")
        except KeyError as e:
            raise ConfigurationError(f"Missing required configuration key: {e}")
