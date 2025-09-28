#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理配置文件
定义常用的耳蜗电图生成参数配置
"""

from typing import Dict, Any


# 默认配置
DEFAULT_CONFIG = {
    'sr': 16000,              # 采样率
    'n_filters': 38,          # 滤波器数量
    'low_lim': 50,            # 低频限制 (Hz)
    'hi_lim': 7500,           # 高频限制 (Hz) - 低于奈奎斯特频率
    'sample_factor': 2,       # 采样因子
    'downsample_factor': None,  # 下采样因子 - 不使用下采样
    'nonlinearity': 'power',  # 非线性变换
    'strict': False,          # 严格模式
    'target_duration': 10.0   # 目标时长 (秒) - 用于标准化时间维度
}

# 语音处理配置
SPEECH_CONFIG = {
    'sr': 16000,              # 语音标准采样率
    'n_filters': 38,          # 标准滤波器数量
    'low_lim': 50,            # 语音低频限制
    'hi_lim': 7500,           # 语音高频限制 (低于奈奎斯特频率)
    'sample_factor': 2,       # 采样因子
    'downsample_factor': 160, # 适度下采样 (16000/160=100)
    'nonlinearity': 'power',  # 幂压缩
    'strict': False,
    'target_duration': 10.0   # 目标时长 (秒)
}

# 音乐处理配置 - 修复版本
MUSIC_CONFIG = {
    'sr': 22050,              # 音乐采样率
    'n_filters': 64,          # 更多滤波器
    'low_lim': 20,            # 音乐低频限制
    'hi_lim': 20000,          # 音乐高频限制
    'sample_factor': 4,       # 高采样因子
    'downsample_factor': 220, # 较少下采样 (22050/220=100.23)
    'nonlinearity': 'db',     # 分贝变换
    'strict': False,
    'target_duration': 10.0   # 目标时长 (秒)
}

# 高质量配置
HIGH_QUALITY_CONFIG = {
    'sr': 44100,              # 高质量采样率
    'n_filters': 64,          # 高分辨率滤波器
    'low_lim': 20,            # 低频限制
    'hi_lim': 20000,          # 高频限制
    'sample_factor': 4,       # 高采样因子
    'downsample_factor': None, # 不下采样
    'nonlinearity': 'power',  # 幂压缩
    'strict': True,           # 严格模式
    'target_duration': 10.0   # 目标时长 (秒)
}

# 64-band配置 (实际输出约133个通道)
COCHLEAGRAM_64BAND_CONFIG = {
    'sr': 16000,              # 标准采样率
    'n_filters': 64,          # 64个滤波器
    'low_lim': 50,            # 低频限制
    'hi_lim': 7500,           # 高频限制（低于奈奎斯特频率）
    'sample_factor': 2,       # 采样因子
    'downsample_factor': None, # 不使用下采样
    'nonlinearity': 'power',  # 非线性变换
    'strict': False,          # 非严格模式
    'target_duration': 10.0   # 目标时长 (秒)
}

# 真正的64通道配置
COCHLEAGRAM_64CHANNEL_CONFIG = {
    'sr': 16000,              # 标准采样率
    'n_filters': 32,          # 32个滤波器
    'low_lim': 50,            # 低频限制
    'hi_lim': 7500,           # 高频限制（低于奈奎斯特频率）
    'sample_factor': 2,       # 采样因子 (32*2=64)
    'downsample_factor': None, # 不使用下采样
    'nonlinearity': 'power',  # 非线性变换
    'strict': False,          # 非严格模式
    'target_duration': 10.0   # 目标时长 (秒)
}

# 高分辨率64通道配置
COCHLEAGRAM_64CHANNEL_HIGH_RES_CONFIG = {
    'sr': 16000,              # 标准采样率
    'n_filters': 64,          # 64个滤波器
    'low_lim': 50,            # 低频限制
    'hi_lim': 7500,           # 高频限制（低于奈奎斯特频率）
    'sample_factor': 1,       # 采样因子 (64*1=64)
    'downsample_factor': None, # 不使用下采样
    'nonlinearity': 'power',  # 非线性变换
    'strict': False,          # 非严格模式
    'target_duration': 10.0   # 目标时长 (秒)
}

# 快速处理配置
FAST_CONFIG = {
    'sr': 8000,               # 低采样率
    'n_filters': 24,          # 较少滤波器
    'low_lim': 100,           # 较高低频限制
    'hi_lim': 3500,           # 较低高频限制 (低于奈奎斯特频率)
    'sample_factor': 1,       # 低采样因子
    'downsample_factor': 80,  # 高下采样 (8000/80=100)
    'nonlinearity': 'power',  # 幂压缩
    'strict': False
}

# 配置字典
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
    获取指定配置
    
    Args:
        config_name: 配置名称
        
    Returns:
        配置字典
    """
    if config_name not in CONFIGS:
        raise ValueError(f"未知的配置名称: {config_name}. 可用配置: {list(CONFIGS.keys())}")
    
    return CONFIGS[config_name].copy()


def list_configs() -> list:
    """
    列出所有可用配置
    
    Returns:
        配置名称列表
    """
    return list(CONFIGS.keys())


def print_config_info():
    """打印配置信息"""
    print("可用的预处理配置:")
    print("=" * 50)
    
    for name, config in CONFIGS.items():
        print(f"\n{name.upper()} 配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print("\n使用示例:")
    print("from config import get_config")
    print("config = get_config('speech')")
    print("preprocessor = CochleagramPreprocessor(**config)")


if __name__ == '__main__':
    print_config_info() 