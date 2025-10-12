#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data preprocessing configuration file
Define common cochleagram generation parameter configurations
"""

from typing import Dict, Any


# Default configuration
DEFAULT_CONFIG = {
    'sr': 16000,              # Sampling rate
    'n_filters': 38,          # Number of filters
    'low_lim': 50,            # Low frequency limit (Hz)
    'hi_lim': 7500,           # High frequency limit (Hz) - below Nyquist frequency
    'sample_factor': 2,       # Sampling factor
    'downsample_factor': None,  # Downsampling factor - no downsampling used
    'nonlinearity': 'power',  # Power compression
    'strict': False,          # Mode
    'target_duration': 10.0   # Target duration (seconds) - used to standardize the temporal dimension of cochlear electrograms
}

# Speech processing configuration
SPEECH_CONFIG = {
    'sr': 16000,              # Standard sampling rate
    'n_filters': 38,          # Number of filters
    'low_lim': 50,            # Low frequency limit (Hz)
    'hi_lim': 7500,           # High frequency limit (Hz) - below Nyquist frequency
    'sample_factor': 2,       # Sampling factor
    'downsample_factor': 160, # Moderate downsampling (16000/160=100)
    'nonlinearity': 'power',  # Power compression
    'strict': False,
    'target_duration': 10.0   # Target duration (seconds) - used to standardize the temporal dimension of cochlear electrograms
}

# Music processing configuration
MUSIC_CONFIG = {
    'sr': 22050,              # Music sampling rate
    'n_filters': 64,          # More filters
    'low_lim': 20,            # Music low frequency limit (Hz)
    'hi_lim': 20000,          # Music high frequency limit (Hz)
    'sample_factor': 4,       # High sampling factor
    'downsample_factor': 220, # Less downsampling (22050/220=100.23)
    'nonlinearity': 'db',     # Decibel transformation
    'strict': False,
    'target_duration': 10.0   # Target duration (seconds) - used to standardize the temporal dimension of cochlear electrograms
}

# High-quality configuration
HIGH_QUALITY_CONFIG = {
    'sr': 44100,              # High-quality sampling rate
    'n_filters': 64,          # High-resolution filters
    'low_lim': 20,            # Low frequency limit (Hz)
    'hi_lim': 20000,          # High frequency limit (Hz)
    'sample_factor': 4,       # High sampling factor
    'downsample_factor': None, # No downsampling
    'nonlinearity': 'power',  # Power compression
    'strict': True,           # Strict mode
    'target_duration': 10.0   # Target duration (seconds) - used to standardize the temporal dimension of cochlear electrograms
}

# 64-band configuration (actual output about 133 channels)
COCHLEAGRAM_64BAND_CONFIG = {
    'sr': 16000,              # Standard sampling rate
    'n_filters': 64,          # 64 filters
    'low_lim': 50,            # Low frequency limit (Hz)
    'hi_lim': 7500,           # High frequency limit (Hz) - below Nyquist frequency
    'sample_factor': 2,       # Sampling factor
    'downsample_factor': None, # No downsampling
    'nonlinearity': 'power',  # Nonlinear transformation
    'strict': False,          # Non-strict mode
    'target_duration': 10.0   # Target duration (seconds)
}

# True 64-channel configuration
COCHLEAGRAM_64CHANNEL_CONFIG = {
    'sr': 16000,              # Standard sampling rate
    'n_filters': 32,          # 32 filters
    'low_lim': 50,            # Low frequency limit (Hz)
    'hi_lim': 7500,           # High frequency limit (Hz) - below Nyquist frequency
    'sample_factor': 2,       # Sampling factor (32*2=64)
    'downsample_factor': None, # No downsampling
    'nonlinearity': 'power',  # Nonlinear transformation
    'strict': False,          # Non-strict mode
    'target_duration': 10.0   # Target duration (seconds)
}

# High-resolution 64-channel configuration
COCHLEAGRAM_64CHANNEL_HIGH_RES_CONFIG = {
    'sr': 16000,              # Standard sampling rate
    'n_filters': 64,          # 64 filters
    'low_lim': 50,            # Low frequency limit (Hz)
    'hi_lim': 7500,           # High frequency limit (Hz) - below Nyquist frequency
    'sample_factor': 1,       # Sampling factor (64*1=64)
    'downsample_factor': None, # No downsampling
    'nonlinearity': 'power',  # Nonlinear transformation
    'strict': False,          # Non-strict mode
    'target_duration': 10.0   # Target duration (seconds)
}

# Fast processing configuration
FAST_CONFIG = {
    'sr': 8000,               # Low sampling rate
    'n_filters': 24,          # Less filters
    'low_lim': 100,           # Higher low frequency limit
    'hi_lim': 3500,           # Lower high frequency limit (below Nyquist frequency)
    'sample_factor': 1,       # Low sampling factor
    'downsample_factor': 80,  # High downsampling (8000/80=100)
    'nonlinearity': 'power',  # Power compression
    'strict': False
}

# Configuration dictionary
CONFIGS = {
    'default': DEFAULT_CONFIG,
    'speech': SPEECH_CONFIG,
    'music': MUSIC_CONFIG,
    'high_quality': HIGH_QUALITY_CONFIG,
    '64band': COCHLEAGRAM_64BAND_CONFIG,
    'fast': FAST_CONFIG
}


def get_config(config_name: str = 'default') -> Dict[str, Any]:
    """
    Get specified configuration
    
    Args:
        config_name: Configuration name
        
    Returns:
        Configuration dictionary
    """
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown configuration name: {config_name}. Available configurations: {list(CONFIGS.keys())}")
    
    return CONFIGS[config_name].copy()


def list_configs() -> list:
    """
    List all available configurations
    
    Returns:
        Configuration name list
    """
    return list(CONFIGS.keys())


def print_config_info():
    """Print configuration information"""
    print("Available preprocessing configurations:")
    print("=" * 50)
    
    for name, config in CONFIGS.items():
        print(f"\n{name.upper()} configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print("\nUsage example:")
    print("from config import get_config")
    print("config = get_config('speech')")
    print("preprocessor = CochleagramPreprocessor(**config)")


if __name__ == '__main__':
    print_config_info() 