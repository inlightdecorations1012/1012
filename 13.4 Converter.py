
#!/usr/bin/env python3
"""
EUR/USD Forex Parquet Converter v13.4 - Enhanced Data Preservation
================================================================

v13.4 Changes:
- Added preserve mode validation with data quality masks
- Dynamic spread filter based on median spread
- Dynamic ATR multipliers for better class balance
- Per-timeframe feature validation
- Enhanced quality reporting with spread/validation statistics
- YAML configuration included in checksums
- Feature cache versioning with YAML hash
- Dynamic version handling from config

Entry point and core conversion logic.
"""

import sys
import time
import gc
import json
import warnings
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import multiprocessing as mp
from dataclasses import asdict
from functools import wraps
import logging
import logging.handlers
import psutil
import yaml
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.csv
import hashlib
import pickle

# Import our modules
from config import (
    ForexConfig, TimeframeConfig, ForexDataError,
    ConfigurationError, DataQualityError, ProcessingConfig
)
from pipeline import (
    PipelineStage, DataValidationStage, FeatureEngineeringStage,
    TargetGenerationStage, NormalizationStage, FeatureRegistry
)
from utils import (
    setup_logging_infrastructure, get_merge_tolerance, fast_hash_dataframe,
    validate_dataframe_integrity, check_memory_availability, estimate_memory_usage,
    setup_colab_environment, clean_forex_dataframe, FeatureCache,
    validate_feature_completeness, cleanup_features_for_export,
    compute_yaml_hash, get_expected_features_for_timeframe, validate_timeframe_features
)
from feature_registry import BASE_FEATURES, FUTURE_FEATURES, COMPLETE_77_FEATURES

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Module logger
logger = logging.getLogger(__name__)

# Public API
__all__ = [
    'ForexParquetConverter',
    'process_single_timeframe',
    'process_timeframe_worker',
    'main'
]

# Check PyArrow version for API compatibility
PYARROW_VERSION = tuple(map(int, pa.__version__.split('.')[:2]))
USE_PYARROW_LEGACY_CSV = PYARROW_VERSION < (15, 0)


# === Skip-if-exists Decorator ===
def skip_if_exists(check_metadata: bool = True):
    """Decorator to skip processing if output already exists with matching metadata.

    v13.4: Updated to use dynamic version from config.

    Args:
        check_metadata: If True, also check data checksum
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[Tuple[Path, Dict]]:
            # Extract config and tf_config from args
            # Expecting: process_single_timeframe(config, tf_config, feature_cache)
            if len(args) >= 2:
                config = args[0]
                tf_config = args[1]

                # Build expected output path
                output_path = Path(config.paths.output)
                data_version = config.data_version  # v13.4: Use from config

                # Check for both single file and partitioned dataset
                single_file = output_path / f"EUR_USD_{tf_config.name}_v{data_version}.parquet"
                partitioned_dir = output_path / f"EUR_USD_{tf_config.name}_v{data_version}_dataset"

                # Check if either exists
                existing_path = None
                if partitioned_dir.exists() and partitioned_dir.is_dir():
                    existing_path = partitioned_dir
                elif single_file.exists():
                    existing_path = single_file

                if existing_path:
                    if check_metadata:
                        # Try to read metadata and check checksum
                        report = {}  # Default report in case of errors
                        try:
                            if existing_path.is_dir():
                                # Partitioned dataset
                                dataset = ds.dataset(existing_path)
                                if dataset.schema.metadata:
                                    metadata_bytes = dataset.schema.metadata.get(b'forex_metadata')
                                    if metadata_bytes:
                                        existing_metadata = json.loads(metadata_bytes.decode())
                                        existing_checksum = existing_metadata.get('data_checksum')

                                        # Compute current data checksum
                                        # For efficiency, just check if source files have changed
                                        input_path = Path(config.paths.input)
                                        current_checksum = compute_input_checksum(
                                            input_path, tf_config,
                                            str(config.config_path) if hasattr(config, 'config_path') else None,
                                            config
                                        )

                                        if existing_checksum and current_checksum == existing_checksum:
                                            logger.info(
                                                f"[SKIP] {tf_config.name} - Output exists with matching checksum"
                                            )
                                            # Load quality report if available
                                            report_path = output_path / f"{tf_config.name}_quality_report_v{data_version}.json"
                                            if report_path.exists():
                                                with open(report_path, 'r') as f:
                                                    report = json.load(f)
                                            else:
                                                report = {'skipped': True, 'reason': 'matching_checksum'}

                                            return existing_path, report
                            else:
                                # Single file - read metadata
                                table = pq.read_table(existing_path)
                                if table.schema.metadata:
                                    metadata_bytes = table.schema.metadata.get(b'forex_metadata')
                                    if metadata_bytes:
                                        existing_metadata = json.loads(metadata_bytes.decode())
                                        # Similar checksum logic as above
                                        pass

                            logger.info(f"Checksum mismatch - reprocessing {tf_config.name}")

                        except Exception as e:
                            logger.warning(f"Error checking existing metadata: {e}")
                    else:
                        # No metadata check - just skip if exists
                        logger.info(f"[SKIP] {tf_config.name} - Output exists (no checksum check)")
                        return existing_path, {'skipped': True, 'reason': 'exists_no_check'}

            # Call original function
            return func(*args, **kwargs)

        return wrapper
    return decorator


def compute_input_checksum(input_path: Path, tf_config: TimeframeConfig,
                          config_path: Optional[str] = None,
                          config: Optional[ForexConfig] = None) -> str:
    """Compute checksum of input files for a timeframe.

    v13.4: Includes YAML configuration in checksum and supports content mode.

    Args:
        input_path: Input directory
        tf_config: Timeframe configuration
        config_path: Optional path to YAML config file
        config: Optional ForexConfig for checksum mode

    Returns:
        Hex checksum string
    """
    hasher = hashlib.sha256()

    # Find all relevant input files
    bid_files = sorted(list(input_path.glob(f"{tf_config.file_pattern}BID*.csv")))
    ask_files = sorted(list(input_path.glob(f"{tf_config.file_pattern}ASK*.csv")))

    all_files = bid_files + ask_files

    # v13.4: Use checksum mode from config
    checksum_mode = 'metadata'  # default
    if config and hasattr(config.processing, 'checksum_mode'):
        checksum_mode = config.processing.checksum_mode

    for file_path in all_files:
        # Always hash file path
        hasher.update(str(file_path).encode())

        if checksum_mode == 'content':
            # v13.4: Sample first 10KB of file content
            try:
                with open(file_path, 'rb') as f:
                    content_sample = f.read(10240)  # 10KB
                    hasher.update(content_sample)
            except Exception as e:
                logger.warning(f"Could not read content from {file_path}: {e}")
                # Fall back to metadata
                hasher.update(str(file_path.stat().st_mtime).encode())
                hasher.update(str(file_path.stat().st_size).encode())
        else:
            # Metadata mode (default)
            hasher.update(str(file_path.stat().st_mtime).encode())
            hasher.update(str(file_path.stat().st_size).encode())

    # Include config parameters that affect output
    config_str = f"{tf_config.frequency}_{tf_config.target_config.multi_horizon}_{tf_config.enabled_features}"
    hasher.update(config_str.encode())

    # v13.4: Include YAML configuration hash
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'rb') as f:
                yaml_content = f.read()
                hasher.update(yaml_content)
        except Exception as e:
            logger.warning(f"Could not hash YAML config: {e}")

    # Include version
    if config:
        hasher.update(config.data_version.encode())
    else:
        hasher.update(b"v13.4")  # fallback

    return hasher.hexdigest()


class ForexParquetConverter:
    """Main converter class orchestrating the ETL pipeline."""

    def __init__(self, config_path: str):
        # ------------------------------------------------------------------
        # 1.  Load YAML configuration
        # ------------------------------------------------------------------
        self.config_path = Path(config_path).resolve()
        self.config = self._load_configuration(str(self.config_path))
        # Store config path in config object for checksum calculation
        self.config.config_path = self.config_path

        # v13.4: Compute YAML hash for cache versioning
        self.yaml_hash = compute_yaml_hash(self.config_path)

        # ------------------------------------------------------------------
        # 2.  Prepare input / output folders
        # ------------------------------------------------------------------
        self.input_path = Path(self.config.paths.input)
        self.output_path = Path(self.config.paths.output)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # 3.  Set-up logging (QueueListener pattern)
        # ------------------------------------------------------------------
        self.log_queue: mp.Queue = mp.Queue()
        self.log_listener = None
        self._setup_logging()

        # ------------------------------------------------------------------
        # 4.  Create the feature registry **and** the cache
        #     (cache MUST exist before the pipeline is built)
        # ------------------------------------------------------------------
        self.feature_registry = FeatureRegistry()
        cache_dir = self.output_path / ".feature_cache"
        # v13.4: Pass YAML hash to cache
        self.feature_cache = FeatureCache(cache_dir, yaml_hash=self.yaml_hash)

        # ------------------------------------------------------------------
        # 5.  Build the pipeline – now it can safely reference self.feature_cache
        # ------------------------------------------------------------------
        self.pipeline = self._build_pipeline()

        # ------------------------------------------------------------------
        # 6.  Build timeframe configs
        # ------------------------------------------------------------------
        self.timeframe_configs = self._build_timeframe_configs()

        # ------------------------------------------------------------------
        # 7.  Sanity checks
        # ------------------------------------------------------------------
        if not self.input_path.exists():
            raise ConfigurationError(f"Input path does not exist: {self.input_path}")

        logger.info(f"Forex Parquet Converter {self.config.data_version} initialized successfully")


    def _load_configuration(self, config_path: str) -> ForexConfig:
        """Load and parse YAML configuration.

        Args:
            config_path: Path to YAML file

        Returns:
            ForexConfig object

        Raises:
            ConfigurationError: If file not found or invalid
        """
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return ForexConfig.from_dict(config_dict)
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML syntax: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def _setup_logging(self) -> None:
        """Setup logging infrastructure with QueueListener pattern."""
        log_config = self.config.logging

        # keep both representations
        level_str = log_config.level                 # e.g. "INFO"
        level_int = getattr(logging, level_str.upper(), logging.INFO)

        # Handlers for the listener
        handlers = []

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level_int)
        handlers.append(console_handler)

        if log_config.save_to_file:
            log_file = self.output_path / f'conversion_v{self.config.data_version}.log'
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(logging.DEBUG)   # extra detail
            handlers.append(file_handler)

        formatter = logging.Formatter(
            '%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        for handler in handlers:
            handler.setFormatter(formatter)

        self.log_listener = logging.handlers.QueueListener(
            self.log_queue, *handlers, respect_handler_level=True
        )
        self.log_listener.start()

        # pass the STRING version into our helper
        setup_logging_infrastructure(self.log_queue, level_str)


    def _build_pipeline(self) -> List[PipelineStage]:
        """Build the processing pipeline stages.

        Returns:
            List of pipeline stages in order
        """
        return [
            DataValidationStage(),
            FeatureEngineeringStage(self.feature_registry, self.feature_cache),
            TargetGenerationStage(),
            NormalizationStage()
        ]

    def _build_timeframe_configs(self) -> Dict[str, TimeframeConfig]:
        """Build timeframe configurations with deep copies.

        Returns:
            Dictionary mapping timeframe names to configs
        """
        import copy
        configs = {}

        for tf_cfg in self.config.timeframes:
            # Deep copy to prevent shared mutable state
            tf_copy = copy.deepcopy(tf_cfg)

            # v13.4: Pass calendar mode from processing config to quality control
            # This allows DataValidationStage to access it
            if hasattr(self.config.processing, 'calendar'):
                # Store calendar mode in a place accessible to validation stage
                tf_copy.quality_control_config.validation_mode = tf_copy.quality_control_config.validation_mode
                # Add as custom attribute (not ideal but works)
                tf_copy._calendar_mode = self.config.processing.calendar

            configs[tf_cfg.name] = tf_copy

        return configs

    def run(self) -> None:
        """Execute the conversion pipeline for all timeframes."""
        logger.info(f"Starting Forex Parquet Converter {self.config.data_version}")
        logger.info(f"Input path: {self.input_path}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Timeframes to process: {list(self.timeframe_configs.keys())}")

        # Check Colab environment
        setup_colab_environment()

        # Check available memory
        available_gb = check_memory_availability()
        logger.info(f"Available memory: {available_gb:.1f} GB")

        # Process based on configuration
        proc_config = self.config.processing

        if proc_config.parallel_enabled and len(self.timeframe_configs) > 1:
            results = self._run_parallel_processing(proc_config)
        else:
            results = self._run_sequential_processing()

        # Generate summary report
        if results:
            self._generate_summary_report(results)
            logger.info(f"Conversion completed! Processed {len(results)} timeframes")
        else:
            logger.warning("No timeframes were processed successfully")

        # Cleanup
        self.feature_cache.cleanup()

        # Shutdown logging
        if self.log_listener:
            self.log_listener.stop()

    def _run_parallel_processing(self, proc_config: 'ProcessingConfig') -> List[Tuple[Path, Dict]]:
        """Run processing in parallel using multiprocessing.

        Args:
            proc_config: Processing configuration

        Returns:
            List of results
        """
        results = []
        max_workers = proc_config.max_workers

        logger.info(f"Using multiprocessing with {max_workers} workers")

        # Use spawn method for better compatibility
        ctx = mp.get_context('spawn')

        with ctx.Pool(processes=max_workers) as pool:
            # Prepare arguments for workers
            worker_args = []
            for tf_cfg in self.timeframe_configs.values():
                # Pass queue and cache directory
                args = (
                    str(self.config_path),
                    tf_cfg.name,
                    self.log_queue,
                    str(self.feature_cache.cache_dir),
                    self.yaml_hash  # v13.4: Pass YAML hash
                )
                worker_args.append(args)

            # Submit all jobs
            async_results = [
                pool.apply_async(process_timeframe_worker, args=args)
                for args in worker_args
            ]

            # Collect results
            for async_result in async_results:
                try:
                    result = async_result.get(timeout=3600)  # 1 hour timeout
                    if result:
                        # Convert string path back to Path
                        output_path, report = result
                        results.append((Path(output_path), report))
                except Exception as e:
                    logger.error(f"Worker process failed: {e}", exc_info=True)

        return results

    def _run_sequential_processing(self) -> List[Tuple[Path, Dict[str, Any]]]:
        """Run processing sequentially.

        Returns:
            List of results
        """
        results = []
        logger.info("Processing timeframes sequentially")

        for tf_config in self.timeframe_configs.values():
            try:
                result = process_single_timeframe(
                    self.config, tf_config, self.feature_cache
                )
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {tf_config.name}: {e}", exc_info=True)

        return results

    def _generate_summary_report(self, results: List[Tuple[Path, Dict]]) -> None:
        """Generate summary report of conversion.

        v13.4: Enhanced with validation statistics and feature coverage.

        Args:
            results: List of (output_path, report) tuples
        """
        summary = {
            'version': self.config.data_version,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_files': len(results),
            'total_size_mb': 0,
            'config_path': str(self.config_path),
            'yaml_hash': self.yaml_hash,
            'pyarrow_version': pa.__version__,
            'validation_mode': self.config.quality_control.validation_mode,
            'results': [],
            'performance_metrics': {
                'cache_hits': self.feature_cache.get_stats()['hits'],
                'cache_misses': self.feature_cache.get_stats()['misses'],
                'cache_hit_rate': self.feature_cache.get_hit_rate()
            }
        }

        # v13.4: Aggregate validation statistics
        total_rows_flagged = 0
        total_rows_processed = 0

        for file_path, report in results:
            # Calculate size (handle both files and directories)
            if file_path.is_dir():
                size_bytes = sum(f.stat().st_size for f in file_path.rglob('*.parquet'))
            else:
                size_bytes = file_path.stat().st_size

            size_mb = size_bytes / (1024 * 1024)
            summary['total_size_mb'] += size_mb

            # v13.4: Extract key statistics
            if 'validation_breakdown' in report:
                total_rows_flagged += report['validation_breakdown'].get('total_flagged', 0)
            total_rows_processed += report.get('rows', 0)

            result_entry = {
                'file': str(file_path),
                'size_mb': round(size_mb, 2),
                'quality_report': report
            }
            summary['results'].append(result_entry)

        # Aggregate statistics
        total_nans = sum(
            r.get('nan_statistics', {}).get('total_nans', 0)
            for _, r in results
        )

        # v13.4: Enhanced aggregate statistics
        summary['aggregate_statistics'] = {
            'total_nans_across_all_files': total_nans,
            'average_file_size_mb': round(summary['total_size_mb'] / len(results), 2) if results else 0,
            'total_rows_processed': total_rows_processed,
            'total_rows_flagged': total_rows_flagged,
            'flagged_percentage': round(total_rows_flagged / total_rows_processed * 100, 2) if total_rows_processed > 0 else 0
        }

        # Save report
        output_file = self.output_path / f'conversion_summary_v{self.config.data_version}.json'
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Summary report saved to {output_file}")


# === Top-level worker function for multiprocessing ===
def process_timeframe_worker(config_path: str, timeframe_name: str,
                           log_queue: mp.Queue, cache_dir: str,
                           yaml_hash: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Worker function for processing a single timeframe in a separate process.

    v13.4: Added yaml_hash parameter for cache versioning.

    This function is picklable and can be safely passed to multiprocessing.

    Args:
        config_path: Path to YAML configuration file
        timeframe_name: Name of timeframe to process
        log_queue: Queue for logging
        cache_dir: Directory for feature cache
        yaml_hash: Hash of YAML configuration

    Returns:
        Tuple of (output_path_str, report_dict) or None on failure
    """
    try:
        # Setup logging for this process
        setup_logging_infrastructure(log_queue, 'INFO', non_blocking=True)
        logger = logging.getLogger(__name__)

        # Load configuration
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = ForexConfig.from_dict(config_dict)
        # Store config path for checksum
        config.config_path = Path(config_path)

        # Find the specific timeframe config
        tf_config = None
        for tf in config.timeframes:
            if tf.name == timeframe_name:
                tf_config = tf
                break

        if not tf_config:
            logger.error(f"Timeframe {timeframe_name} not found in config")
            return None

        # Create feature cache for this process
        feature_cache = FeatureCache(Path(cache_dir), yaml_hash=yaml_hash)

        # Process the timeframe
        result = process_single_timeframe(config, tf_config, feature_cache)

        if result:
            # Convert Path to string for pickling
            output_path, report = result
            return str(output_path), report

        return None

    except Exception as e:
        logger.error(f"Worker failed for {timeframe_name}: {e}", exc_info=True)
        return None


# === Main processing function (decorated with skip_if_exists) ===
@skip_if_exists(check_metadata=True)
def process_single_timeframe(config: ForexConfig,
                           tf_config: TimeframeConfig,
                           feature_cache: Optional[FeatureCache] = None) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """Process a single timeframe through the full pipeline.

    v13.4: Enhanced with per-timeframe feature validation and dynamic versioning.

    Args:
        config: Main configuration
        tf_config: Timeframe-specific configuration
        feature_cache: Optional feature cache for incremental enrichment

    Returns:
        Tuple of (output_path, report_dict) on success, None on failure
    """
    logger.info(f"--- [START] Processing {tf_config.name} timeframe ---")

    try:
        # Initialize components
        feature_registry = FeatureRegistry()
        pipeline = [
            DataValidationStage(),
            FeatureEngineeringStage(feature_registry, feature_cache),
            TargetGenerationStage(),
            NormalizationStage()
        ]

        # Find and read files
        input_path = Path(config.paths.input)
        df = read_and_merge_timeframe_data(input_path, tf_config, config.processing)

        if df is None or df.empty:
            logger.warning(f"No valid data for {tf_config.name}")
            return None

        # Apply debug row limit if specified
        if tf_config.debug_config.limit_rows:
            logger.warning(f"DEBUG: Limiting to {tf_config.debug_config.limit_rows} rows")
            df = df.tail(tf_config.debug_config.limit_rows).reset_index(drop=True)

        # Run pipeline with memory checks after each stage
        for stage in pipeline:
            logger.info(f"[{tf_config.name}] Running {stage.name}")

            # Check memory before and after each stage
            memory_before = df.memory_usage(deep=True).sum() / (1024 * 1024)
            df = stage.transform(df, tf_config)
            memory_after = df.memory_usage(deep=True).sum() / (1024 * 1024)

            logger.info(
                f"[{tf_config.name}] After {stage.name}: "
                f"{memory_before:.1f} MB → {memory_after:.1f} MB"
            )

            # Check if memory is growing too much
            if memory_after > memory_before * 2.5:
                logger.warning(
                    f"[{tf_config.name}] Large memory growth in {stage.name}: "
                    f"{memory_before:.1f} MB → {memory_after:.1f} MB"
                )

        # v13.4: Validate against expected features for this timeframe
expected_features = get_expected_features_for_timeframe(tf_config, feature_registry)

# Exclude validation columns from feature validation
feature_cols_for_validation = [
    col for col in df.columns 
    if col not in ['is_valid_price', 'is_valid_spread', 'is_valid_time_gap']
]
validate_feature_completeness(
    df[feature_cols_for_validation], 
    list(expected_features), 
    FUTURE_FEATURES
)

        # Clean up features before export
        df = cleanup_features_for_export(df, COMPLETE_77_FEATURES)

        # v13.4: Explicitly drop unexpected columns
        expected_columns = (
            ['timestamp', 'data_validity_mask'] +
            list(expected_features) +
            [col for col in df.columns if col.startswith('target')] +
            [f'{f}_norm' for f in expected_features if f'{f}_norm' in df.columns]
        )

        # Add validation masks if in preserve mode
        if tf_config.quality_control_config.validation_mode == 'preserve':
            expected_columns.extend(['is_valid_price', 'is_valid_spread', 'is_valid_time_gap'])

        # Drop any columns not in expected list
        unexpected_cols = [col for col in df.columns if col not in expected_columns]
        if unexpected_cols:
            logger.info(f"Dropping unexpected columns: {unexpected_cols}")
            df = df.drop(columns=unexpected_cols)

        # Generate quality report
        report = generate_quality_report(df, tf_config, feature_registry)

        # Save quality report separately for skip-if-exists
        output_path = Path(config.paths.output)
        report_path = output_path / f"{tf_config.name}_quality_report_v{config.data_version}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Export to parquet
        output_file = export_to_parquet_dataset(df, tf_config, output_path, config)

        logger.info(
            f"--- [SUCCESS] {tf_config.name}: {len(df)} rows, "
            f"{len(df.columns)} columns ---"
        )

        # Cleanup
        del df
        gc.collect()

        return output_file, report

    except Exception as e:
        logger.error(f"[ERROR] Failed processing {tf_config.name}: {e}", exc_info=True)
        return None


def read_and_merge_timeframe_data(input_path: Path,
                                tf_config: TimeframeConfig,
                                proc_config: 'ProcessingConfig') -> Optional[pd.DataFrame]:
    """Read and merge BID/ASK data for a timeframe with true streaming.

    v13.4: Added proc_config parameter for merge tolerance.

    Args:
        input_path: Input directory path
        tf_config: Timeframe configuration
        proc_config: Processing configuration

    Returns:
        Merged dataframe or None if no data
    """
    # Find files
    bid_files = sorted(list(input_path.glob(f"{tf_config.file_pattern}BID*.csv")))
    ask_files = sorted(list(input_path.glob(f"{tf_config.file_pattern}ASK*.csv")))

    if not bid_files or not ask_files:
        logger.warning(
            f"No BID/ASK files found for pattern '{tf_config.file_pattern}' "
            f"in {input_path}"
        )
        return None

    logger.info(
        f"Found {len(bid_files)} BID and {len(ask_files)} ASK files "
        f"for {tf_config.name}"
    )

    # Read files using streaming approach
    bid_df = read_forex_files_streaming(bid_files, 'bid', tf_config.chunk_size)
    ask_df = read_forex_files_streaming(ask_files, 'ask', tf_config.chunk_size)

    if bid_df.empty or ask_df.empty:
        return None

    # Remove duplicates
    bid_df = bid_df.drop_duplicates(subset=['timestamp'])
    ask_df = ask_df.drop_duplicates(subset=['timestamp'])

    # v13.4: Use configured merge tolerance
    tolerance = get_merge_tolerance(tf_config.frequency, config=proc_config)

    logger.info(
        f"[{tf_config.name}] Using merge tolerance: {tolerance} "
        f"(multiplier: {proc_config.merge_tolerance_multiplier})"
    )

    # Two-pass merge to handle timing mismatches
    # First pass: backward merge (default behavior)
    primary_merge = pd.merge_asof(
        bid_df.sort_values('timestamp'),
        ask_df.sort_values('timestamp'),
        on='timestamp',
        direction='backward',
        tolerance=tolerance
    )

    # Second pass: forward merge for rows with NaN ask data
    ask_cols = [c for c in ask_df.columns if c != 'timestamp']
    missing_mask = primary_merge[ask_cols].isna().any(axis=1)

    if missing_mask.any():
        # Get the bid rows that didn't match
        missing_bid_df = bid_df[bid_df['timestamp'].isin(
            primary_merge.loc[missing_mask, 'timestamp']
        )].sort_values('timestamp')

        # Try forward merge
        merged_forward = pd.merge_asof(
            missing_bid_df,
            ask_df.sort_values('timestamp'),
            on='timestamp',
            direction='forward',
            tolerance=tolerance
        )

        # Update the primary merge with successful forward matches
        for idx in primary_merge[missing_mask].index:
            timestamp = primary_merge.loc[idx, 'timestamp']
            forward_match = merged_forward[merged_forward['timestamp'] == timestamp]
            if not forward_match.empty and not forward_match[ask_cols].isna().any(axis=1).iloc[0]:
                for col in ask_cols:
                    primary_merge.loc[idx, col] = forward_match[col].iloc[0]

    # Log merge statistics
    total_rows = len(bid_df)
    matched_rows = len(primary_merge.dropna())
    match_rate = matched_rows / total_rows * 100 if total_rows > 0 else 0

    logger.info(
        f"[{tf_config.name}] Merge complete: "
        f"{matched_rows}/{total_rows} rows matched ({match_rate:.1f}%)"
    )

    # Drop rows where merge failed (no ask data)
    if match_rate < 95:
        logger.warning(
            f"[{tf_config.name}] Low match rate ({match_rate:.1f}%). "
            f"Check time alignment between BID/ASK files."
        )

    # Final cleanup
    df = primary_merge.dropna()

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"[{tf_config.name}] Final merged data: {len(df)} rows")

    return df

def parse_horizon_from_column(col: str) -> Optional[int]:
    """Extract the horizon integer from a column name like 'target_h5_soft'.
    
    Returns the integer horizon or None if not found.
    """
    import re
    match = re.search(r'target_h(\d+)(?:_soft)?', col)
    if match:
        return int(match.group(1))
    return None

def read_forex_files_streaming(file_paths: List[Path], suffix: str,
                              chunk_size: int = 50000) -> pd.DataFrame:
    """Read multiple forex CSV files using true streaming approach.

    Uses PyArrow for efficient streaming reads to minimize memory usage.

    Args:
        file_paths: List of CSV file paths
        suffix: Suffix for column names ('bid' or 'ask')
        chunk_size: Rows per chunk for streaming

    Returns:
        Combined dataframe
    """
    all_data = []
    total_rows = 0

    for file_path in file_paths:
        try:
            logger.debug(f"Reading {file_path.name}")

            # Column types for the CSV
            column_types = {
                'timestamp': pa.timestamp('ns'),
                'open': pa.float64(),
                'high': pa.float64(),
                'low': pa.float64(),
                'close': pa.float64(),
                'volume': pa.float64()
            }

            # Read options
            read_options = pyarrow.csv.ReadOptions(
                use_threads=True,
                block_size=chunk_size * 1024  # Approximate
            )

            # Parse options
            parse_options = pyarrow.csv.ParseOptions(
                delimiter=','
            )

            # Convert options
            convert_options = pyarrow.csv.ConvertOptions(
                column_types=column_types,
                timestamp_parsers=['%Y-%m-%d %H:%M:%S'],
                include_columns=None,  # Read all columns
                include_missing_columns=True,
                strings_can_be_null=True
            )

            # Use appropriate CSV reading method based on PyArrow version
            if USE_PYARROW_LEGACY_CSV:
                # Legacy approach for older PyArrow
                table = pyarrow.csv.read_csv(
                    file_path,
                    read_options=read_options,
                    parse_options=parse_options,
                    convert_options=convert_options
                )

                # Convert to pandas
                df = table.to_pandas()

                # Process in chunks if large
                if len(df) > chunk_size:
                    for i in range(0, len(df), chunk_size):
                        chunk_df = df.iloc[i:i+chunk_size]
                        cleaned_chunk = clean_forex_dataframe(chunk_df, suffix)

                        if not cleaned_chunk.empty:
                            all_data.append(cleaned_chunk)
                            total_rows += len(cleaned_chunk)
                else:
                    cleaned_df = clean_forex_dataframe(df, suffix)
                    if not cleaned_df.empty:
                        all_data.append(cleaned_df)
                        total_rows += len(cleaned_df)

            else:
                # Modern streaming approach
                with pyarrow.csv.open_csv(
                    file_path,
                    read_options=read_options,
                    parse_options=parse_options,
                    convert_options=convert_options
                ) as reader:

                    for batch in reader:
                        # Convert batch to pandas
                        chunk_df = batch.to_pandas()

                        # Clean and append
                        cleaned_chunk = clean_forex_dataframe(chunk_df, suffix)

                        if not cleaned_chunk.empty:
                            all_data.append(cleaned_chunk)
                            total_rows += len(cleaned_chunk)

            logger.debug(f"Read {total_rows} rows from {file_path.name}")

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # Final deduplication
        return final_df.drop_duplicates('timestamp').sort_values('timestamp')
    else:
        return pd.DataFrame()



def generate_quality_report(df: pd.DataFrame,
                          tf_config: TimeframeConfig,
                          feature_registry: 'FeatureRegistry') -> Dict[str, Any]:
    """Generate comprehensive quality report.

    v13.4: Enhanced with spread statistics, validation breakdown, and feature coverage.

    Args:
        df: Processed dataframe
        tf_config: Timeframe configuration
        feature_registry: Feature registry for validation

    Returns:
        Quality report dictionary
    """
    report = {
        'timeframe': tf_config.name,
        'frequency': tf_config.frequency,
        'rows': len(df),
        'columns': len(df.columns),
        'date_range': {
            'start': str(df['timestamp'].min()) if 'timestamp' in df else None,
            'end': str(df['timestamp'].max()) if 'timestamp' in df else None,
            'days': (df['timestamp'].max() - df['timestamp'].min()).days if 'timestamp' in df else None
        },
        'feature_count': len([c for c in df.columns if not c.startswith('target') and c != 'timestamp']),
        'target_horizons': tf_config.target_config.multi_horizon
    }

    # v13.4: Spread statistics
    if 'bid_ask_spread' in df.columns:
        report['spread_statistics'] = {
            'mean': float(df['bid_ask_spread'].mean()),
            'median': float(df['bid_ask_spread'].median()),
            'std': float(df['bid_ask_spread'].std()),
            'min': float(df['bid_ask_spread'].min()),
            'max': float(df['bid_ask_spread'].max()),
            'percentiles': {
                '25%': float(df['bid_ask_spread'].quantile(0.25)),
                '50%': float(df['bid_ask_spread'].quantile(0.50)),
                '75%': float(df['bid_ask_spread'].quantile(0.75)),
                '95%': float(df['bid_ask_spread'].quantile(0.95)),
                '99%': float(df['bid_ask_spread'].quantile(0.99))
            }
        }

    # v13.4: Validation breakdown (if in preserve mode)
    if 'is_valid_price' in df.columns:
        report['validation_breakdown'] = {
            'invalid_price_rows': int((~df['is_valid_price']).sum()),
            'invalid_spread_rows': int((~df['is_valid_spread']).sum()) if 'is_valid_spread' in df else 0,
            'invalid_time_gap_rows': int((~df['is_valid_time_gap']).sum()) if 'is_valid_time_gap' in df else 0,
            'total_flagged': int((~(df['is_valid_price'] &
                                   df.get('is_valid_spread', True) &
                                   df.get('is_valid_time_gap', True))).sum())
        }

    # NaN statistics
    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0].to_dict()

    report['nan_statistics'] = {
        'columns_with_nans': len(nan_counts),
        'total_nans': sum(nan_counts.values()),
        'nan_by_column': nan_counts
    }

    # Target class distribution
    target_cols = [c for c in df.columns if c.startswith('target_h') and c.endswith('_soft')]
    if target_cols:
        class_dist = {}
        for col in target_cols:
            horizon = parse_horizon_from_column(col)
            if horizon and col in df.columns:
                dist = df[col].value_counts(normalize=True).sort_index().to_dict()
                # Convert numpy types to Python types for JSON serialization
                dist = {int(k): float(v) for k, v in dist.items()}
                class_dist[f'h{horizon}'] = dist

        report['target_class_distribution'] = class_dist

        # v13.4: Check if minority class targets were achieved
        minority_targets = {}
        target_pct = tf_config.target_config.target_minority_class_pct
        for horizon_key, dist in class_dist.items():
            if -1 in dist and 1 in dist:
                minority_pct = min(dist.get(-1, 0), dist.get(1, 0))
                minority_targets[horizon_key] = {
                    'achieved': float(minority_pct),
                    'target': float(target_pct),
                    'met': minority_pct >= target_pct * 0.9  # 90% of target is acceptable
                }
        report['minority_class_targets'] = minority_targets

    # Feature validation result
    if hasattr(df, 'attrs') and 'feature_validation' in df.attrs:
        report['feature_validation'] = df.attrs['feature_validation']
    else:
        # Run validation if not already done
        validation_result = validate_timeframe_features(df, tf_config, feature_registry)
        report['feature_validation'] = validation_result

    # Memory usage
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    report['memory_usage_mb'] = round(memory_usage_mb, 2)

    # Data quality metrics
    if 'data_validity_mask' in df.columns:
        validity_mean = df['data_validity_mask'].mean()
        report['data_quality'] = {
            'validity_score': float(validity_mean),
            'invalid_rows': int((df['data_validity_mask'] < 1.0).sum()),
            'invalid_percentage': float((1 - validity_mean) * 100)
        }

    return report


def export_to_parquet_dataset(df: pd.DataFrame, tf_config: TimeframeConfig,
                            output_path: Path, config: ForexConfig) -> Path:
    """Export DataFrame to Parquet with metadata and optional partitioning.

    v13.4: Updated to use dynamic version from config.

    Args:
        df: DataFrame to export
        tf_config: Timeframe configuration
        output_path: Output directory
        config: Main configuration

    Returns:
        Path to output file/directory
    """
    # 1) down-cast FLOAT64 & INT64
    float64_cols = df.select_dtypes(include=["float64"]).columns
    if len(float64_cols):
        mins, maxs = df[float64_cols].min(), df[float64_cols].max()
        if (mins > np.finfo(np.float32).min).all() and (maxs < np.finfo(np.float32).max).all():
            df[float64_cols] = df[float64_cols].astype(np.float32)

    for col in df.select_dtypes(include=["int64"]).columns:
        cmin, cmax = df[col].min(), df[col].max()
        if cmin >= 0:
            if cmax <= np.iinfo(np.uint8).max:
                df[col] = df[col].astype(np.uint8)
            elif cmax <= np.iinfo(np.uint16).max:
                df[col] = df[col].astype(np.uint16)
            elif cmax <= np.iinfo(np.uint32).max:
                df[col] = df[col].astype(np.uint32)
        else:
            if np.iinfo(np.int8).min  <= cmin and cmax <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif np.iinfo(np.int16).min <= cmin and cmax <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif np.iinfo(np.int32).min <= cmin and cmax <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

    # 2) build metadata
    config_path = str(config.config_path) if hasattr(config, 'config_path') else None
    input_checksum = compute_input_checksum(Path(config.paths.input), tf_config, config_path, config)
    meta = {
        "data_version": config.data_version,
        "timeframe": tf_config.name,
        "features_list": sorted([c for c in df.columns if not c.startswith("target") and c != "timestamp"]),
        "feature_count": len([c for c in df.columns if not c.startswith("target") and c != "timestamp"]),
        "original_feature_count": len(
            [c for c in df.columns if not c.startswith("target") and c != "timestamp" and not c.endswith("_norm")]
        ),
        "target_info": {
            "type": tf_config.target_config.type,
            "horizons": tf_config.target_config.multi_horizon,
            "atr_multiplier": tf_config.target_config.atr_multiplier,
            "dynamic_atr_enabled": tf_config.target_config.dynamic_atr_multiplier,
        },
        "normalization_stats": df.attrs.get("normalization_stats", {}),
        "data_checksum": input_checksum,
        "validation_mode": tf_config.quality_control_config.validation_mode,
    }

    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"forex_metadata": json.dumps(meta).encode()}
    )

    # 3) Choose partitioned dataset or single .parquet
    use_partitioning = getattr(config.processing, "partitioning_enabled", True)

    if use_partitioning and "timestamp" in df.columns and len(df) > 10_000:
        out_dir = output_path / f"EUR_USD_{tf_config.name}_v{config.data_version}_dataset"
        out_dir.mkdir(parents=True, exist_ok=True)

        df_part = df.copy()
        df_part["year"]  = df["timestamp"].dt.year
        df_part["month"] = df["timestamp"].dt.month
        table_part = pa.Table.from_pandas(df_part, preserve_index=False)
        table_part = table_part.replace_schema_metadata(table.schema.metadata)

        pq.write_to_dataset(
            table_part,
            root_path=str(out_dir),
            partition_cols=["year", "month"],
            compression="snappy",
            existing_data_behavior="overwrite_or_ignore",
        )

        logger.info(f"Exported partitioned dataset to {out_dir}")
        return out_dir
    else:
        out_file = output_path / f"EUR_USD_{tf_config.name}_v{config.data_version}.parquet"
        pq.write_table(table, str(out_file), compression="snappy")
        logger.info(f"Exported single file to {out_file}")
        return out_file


# === Entry Point ===
def main():
    """Main entry point for the converter."""
    import argparse

    parser = argparse.ArgumentParser(
        description='EUR/USD Forex Parquet Converter v13.4'
    )
    parser.add_argument(
        'config_path',
        type=str,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 13.4'
    )

    args = parser.parse_args()

    try:
        # Create and run converter
        converter = ForexParquetConverter(args.config_path)
        converter.run()

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
