#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于pycochleagram库的数据预处理流程
将.wav格式的音频转换为耳蜗电图并保存
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')

from pycochleagram import cochleagram as cgram
from pycochleagram import erbfilter as erb
from pycochleagram import utils


class CochleagramPreprocessor:
    """耳蜗电图数据预处理器"""
    
    def __init__(self, 
                 sr: int = 16000,
                 n_filters: int = 64,
                 low_lim: int = 50,
                 hi_lim: int = 20000,
                 sample_factor: int = 2,
                 downsample_factor: Optional[int] = None,
                 nonlinearity: Optional[str] = 'power',
                 strict: bool = False,
                 max_duration: float = 10.0,
                 target_duration: Optional[float] = None,
                 ihc_lowpass_cutoff: float = 3000.0,
                 ihc_lowpass_order: int = 7,
                 align_window_2_to_5: bool = False,
                 post_rectify: bool = True,
                 power_point_three: bool = True):
        """
        初始化预处理器
        
        Args:
            sr: 采样率
            n_filters: 滤波器数量 (默认64)
            low_lim: 低频限制 (Hz)
            hi_lim: 高频限制 (Hz)
            sample_factor: 采样因子 (1, 2, 4)
            downsample_factor: 下采样因子，None表示不下采样
            nonlinearity: 非线性变换类型 ('db', 'power', None)
            strict: 是否使用严格模式
            max_duration: 最大音频时长 (秒)
            target_duration: 目标音频时长 (秒)，用于标准化耳蜗电图时间维度
            ihc_lowpass_cutoff: IHC低通滤波器截止频率 (Hz，默认3000Hz)
            ihc_lowpass_order: IHC低通滤波器阶数 (默认7阶)
        """
        self.sr = sr
        self.n_filters = n_filters
        self.low_lim = low_lim
        self.hi_lim = hi_lim
        self.sample_factor = sample_factor
        self.downsample_factor = downsample_factor
        self.nonlinearity = nonlinearity
        self.strict = strict
        self.max_duration = max_duration
        self.target_duration = target_duration
        self.ihc_lowpass_cutoff = ihc_lowpass_cutoff
        self.ihc_lowpass_order = ihc_lowpass_order
        self.align_window_2_to_5 = align_window_2_to_5
        self.post_rectify = post_rectify
        self.power_point_three = power_point_three
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            (signal, sr): 音频信号和采样率
        """
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
            
        # 使用pycochleagram的utils加载音频
        signal, sr = utils.wav_to_array(audio_path)
        
        # 保留多声道；后续在生成耳蜗图时分别处理左右声道
        if len(signal.shape) > 1 and signal.shape[1] > 1:
            self.logger.info(f"检测到立体声音频，将分别处理左右声道 (形状: {signal.shape})")
        
        # 如果需要重采样
        if sr != self.sr:
            self.logger.info(f"重采样从 {sr}Hz 到 {self.sr}Hz")
            # 这里可以添加重采样逻辑
            # 暂时使用简单的下采样
            if sr > self.sr:
                factor = sr // self.sr
                signal = signal[::factor]
                sr = self.sr
        
        # 截断音频到指定时长（支持单/多声道）
        max_samples = int(self.max_duration * sr)
        num_samples = signal.shape[0]
        if num_samples > max_samples:
            self.logger.info(f"截断音频从 {num_samples} 样本到 {max_samples} 样本 (时长: {self.max_duration}秒)")
            signal = signal[:max_samples, ...] if signal.ndim > 1 else signal[:max_samples]
        else:
            self.logger.info(f"音频时长: {num_samples / sr:.2f}秒 (未超过限制 {self.max_duration}秒)")

        # 若需要对齐到固定2-5秒三秒窗口
        if self.align_window_2_to_5:
            need_total = int(5.0 * sr)
            cur_len = signal.shape[0]
            if cur_len < need_total:
                pad = need_total - cur_len
                if signal.ndim == 1:
                    signal = np.pad(signal, (0, pad), mode='constant', constant_values=0)
                else:
                    signal = np.pad(signal, ((0, pad), (0, 0)), mode='constant', constant_values=0)
                self.logger.info(f"为2-5秒窗口填充到5秒: pad={pad} 样本")
            start = int(2.0 * sr)
            end = int(5.0 * sr)
            signal = signal[start:end, ...] if signal.ndim > 1 else signal[start:end]
            self.logger.info(f"已裁剪到固定窗口 [2s, 5s): {signal.shape[0]} 样本")
        else:
            # 如果指定了目标时长，进行标准化处理
            if self.target_duration is not None:
                target_samples = int(self.target_duration * sr)
                self.logger.info(f"目标时长: {self.target_duration}秒, 目标样本数: {target_samples}")
                cur_len = signal.shape[0]
                if cur_len < target_samples:
                    # 填充零（支持多声道）
                    padding = target_samples - cur_len
                    if signal.ndim == 1:
                        signal = np.pad(signal, (0, padding), mode='constant', constant_values=0)
                    else:
                        signal = np.pad(signal, ((0, padding), (0, 0)), mode='constant', constant_values=0)
                    self.logger.info(f"填充音频从 {cur_len} 样本到 {target_samples} 样本")
                elif cur_len > target_samples:
                    # 截断（支持多声道）
                    signal = signal[:target_samples, ...] if signal.ndim > 1 else signal[:target_samples]
                    self.logger.info(f"截断音频从 {cur_len} 样本到 {target_samples} 样本")
                else:
                    self.logger.info(f"音频长度已匹配目标长度: {signal.shape[0]} 样本")
        
        return signal, sr
    
    def generate_cochleagram(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """
        生成耳蜗电图
        
        Args:
            signal: 音频信号
            sr: 采样率
            
        Returns:
            cochleagram: 耳蜗电图数组
        """
        # 移除耳蜗电图生成开始日志
        
        # 动态调整高频限制，避免超过奈奎斯特频率
        nyquist_freq = sr // 2
        adjusted_hi_lim = min(self.hi_lim, nyquist_freq)
        if adjusted_hi_lim != self.hi_lim:
            self.logger.info(f"调整高频限制从 {self.hi_lim}Hz 到 {adjusted_hi_lim}Hz (奈奎斯特频率: {nyquist_freq}Hz)")
        
        # 记录IHC低通滤波器参数用于phase locking控制
        # self.logger.info(f"IHC低通滤波器参数: 截止频率={self.ihc_lowpass_cutoff}Hz, 阶数={self.ihc_lowpass_order}")
        
        # 智能调整下采样因子，确保整数倍关系
        adjusted_downsample = self.downsample_factor
        if adjusted_downsample is not None:
            # 对于pycochleagram的poly模式，downsample参数被解释为env_sr（下采样后的采样率）
            # 需要确保audio_sr/env_sr是整数
            audio_sr = sr
            env_sr = adjusted_downsample  # 这里adjusted_downsample就是env_sr
            
            # 检查是否为整数倍关系
            if audio_sr % env_sr != 0:
                # 找到最接近的整数env_sr，确保audio_sr能被整除
                # 计算下采样因子
                downsample_factor = audio_sr // env_sr
                target_env_sr = audio_sr // downsample_factor
                
                if target_env_sr != env_sr:
                    self.logger.info(f"调整下采样因子从 {adjusted_downsample} 到 {target_env_sr} (audio_sr: {audio_sr}, 下采样因子: {downsample_factor})")
                    adjusted_downsample = target_env_sr
        
        # 生成耳蜗电图（支持单/双声道）。若为立体声，分别处理左右声道并在最后一维堆叠为2通道。
        if signal.ndim == 2 and signal.shape[1] >= 2:
            left = signal[:, 0]
            right = signal[:, 1]

            coch_L = cgram.human_cochleagram(
                left,
                sr,
                n=self.n_filters,
                low_lim=self.low_lim,
                hi_lim=adjusted_hi_lim,
                sample_factor=self.sample_factor,
                downsample=adjusted_downsample,
                nonlinearity=self.nonlinearity,
                strict=self.strict
            )
            coch_R = cgram.human_cochleagram(
                right,
                sr,
                n=self.n_filters,
                low_lim=self.low_lim,
                hi_lim=adjusted_hi_lim,
                sample_factor=self.sample_factor,
                downsample=adjusted_downsample,
                nonlinearity=self.nonlinearity,
                strict=self.strict
            )

            coch_L = np.flipud(coch_L)
            coch_R = np.flipud(coch_R)

            coch_L = self._apply_bez2018_phase_locking(coch_L, sr)
            coch_R = self._apply_bez2018_phase_locking(coch_R, sr)

            if self.post_rectify:
                coch_L = np.maximum(coch_L, 0.0)
                coch_R = np.maximum(coch_R, 0.0)
            if self.power_point_three:
                coch_L = np.power(coch_L, 0.3)
                coch_R = np.power(coch_R, 0.3)

            # 最后一维为声道数: [F, T, 2]
            coch = np.stack([coch_L, coch_R], axis=-1)
            return coch
        else:
            coch = cgram.human_cochleagram(
                signal,
                sr,
                n=self.n_filters,
                low_lim=self.low_lim,
                hi_lim=adjusted_hi_lim,
                sample_factor=self.sample_factor,
                downsample=adjusted_downsample,
                nonlinearity=self.nonlinearity,
                strict=self.strict
            )

            coch = np.flipud(coch)
            coch = self._apply_bez2018_phase_locking(coch, sr)

            if self.post_rectify:
                coch = np.maximum(coch, 0.0)
            if self.power_point_three:
                coch = np.power(coch, 0.3)

            return coch
    
    def save_cochleagram(self, cochleagram: np.ndarray, output_path: str, 
                        save_format: str = 'npy') -> None:
        """
        保存耳蜗电图
        
        Args:
            cochleagram: 耳蜗电图数组
            output_path: 输出路径
            save_format: 保存格式 ('npy', 'npz', 'png')
        """
        self.logger.info(f"保存耳蜗电图到: {output_path}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:  # 只有当路径包含目录时才创建
            os.makedirs(output_dir, exist_ok=True)
        
        if save_format == 'npy':
            np.save(output_path, cochleagram)
        elif save_format == 'npz':
            np.savez_compressed(output_path, cochleagram=cochleagram)
        elif save_format == 'png':
            self._save_as_image(cochleagram, output_path)
        else:
            raise ValueError(f"不支持的保存格式: {save_format}")
            
        self.logger.info(f"耳蜗电图已保存: {output_path}")
    
    def _apply_bez2018_phase_locking(self, cochleagram: np.ndarray, sr: int) -> np.ndarray:
        """
        基于BEZ2018模型的phase locking控制
        
        根据文献描述，通过调整IHC低通滤波器的截止频率来控制phase locking上限。
        在未修改的听觉神经模型中，IHC膜电位的低通特性被建模为截止频率为3000Hz的7阶滤波器。
        
        Args:
            cochleagram: 耳蜗电图数组
            sr: 采样率
            
        Returns:
            处理后的耳蜗电图
        """
        # self.logger.info(f"应用BEZ2018 phase locking控制: IHC低通滤波器截止频率={self.ihc_lowpass_cutoff}Hz, 阶数={self.ihc_lowpass_order}")
        
        # 计算每个滤波器对应的中心频率
        n_filters = cochleagram.shape[0]
        freqs = np.logspace(np.log10(self.low_lim), np.log10(self.hi_lim), n_filters)
        
        # 应用基于IHC低通滤波器的phase locking控制
        for i, freq in enumerate(freqs):
            # 对于高于IHC低通滤波器截止频率的频率，应用phase locking限制
            if freq > self.ihc_lowpass_cutoff:
                # 计算phase locking衰减因子
                # 基于BEZ2018模型的IHC低通滤波器特性
                attenuation_factor = self._calculate_phase_locking_attenuation(freq)
                
                # 应用衰减
                cochleagram[i, :] = cochleagram[i, :] * attenuation_factor
                
                self.logger.debug(f"滤波器 {i} (中心频率: {freq:.1f}Hz) 应用phase locking衰减: {attenuation_factor:.3f}")
        
        return cochleagram
    
    def _calculate_phase_locking_attenuation(self, frequency: float) -> float:
        """
        计算基于BEZ2018模型的phase locking衰减因子
        
        Args:
            frequency: 频率 (Hz)
            
        Returns:
            衰减因子 (0-1之间)
        """
        # 基于BEZ2018模型的IHC低通滤波器特性
        # 使用Butterworth低通滤波器的频率响应特性
        cutoff_freq = self.ihc_lowpass_cutoff
        order = self.ihc_lowpass_order
        
        # 计算归一化频率
        normalized_freq = frequency / cutoff_freq
        
        # 计算Butterworth低通滤波器的幅度响应
        # |H(f)| = 1 / sqrt(1 + (f/fc)^(2n))
        magnitude_response = 1.0 / np.sqrt(1.0 + (normalized_freq ** (2 * order)))
        
        # 对于phase locking，我们使用更严格的衰减
        # 根据文献，phase locking在1500Hz以上开始显著下降
        if frequency > 1500:
            # 额外的phase locking衰减
            phase_lock_attenuation = np.exp(-(frequency - 1500) / 1000)
            magnitude_response *= phase_lock_attenuation
        
        return magnitude_response
    
    def _save_as_image(self, cochleagram: np.ndarray, output_path: str) -> None:
        """保存为图像文件（支持单/双声道）。"""
        if cochleagram.ndim == 2:
            plt.figure(figsize=(10, 6))
            utils.cochshow(cochleagram, interact=False)
            plt.title('Cochleagram')
            plt.ylabel('Filter #')
            plt.xlabel('Time')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        elif cochleagram.ndim == 3 and cochleagram.shape[-1] == 2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
            utils.cochshow(cochleagram[..., 0], ax=axes[0], interact=False)
            axes[0].set_title('Cochleagram - Left')
            axes[0].set_ylabel('Filter #')
            axes[0].set_xlabel('Time')
            axes[0].invert_yaxis()
            utils.cochshow(cochleagram[..., 1], ax=axes[1], interact=False)
            axes[1].set_title('Cochleagram - Right')
            axes[1].set_xlabel('Time')
            axes[1].invert_yaxis()
            fig.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            # 回退到展示第一个通道，避免异常
            plt.figure(figsize=(10, 6))
            to_show = cochleagram[..., 0] if cochleagram.ndim == 3 else cochleagram
            utils.cochshow(to_show, interact=False)
            plt.title('Cochleagram (first channel)')
            plt.ylabel('Filter #')
            plt.xlabel('Time')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    def process_single_file(self, input_path: str, output_path: str, 
                           save_format: str = 'npy') -> Dict[str, Any]:
        """
        处理单个音频文件
        
        Args:
            input_path: 输入音频文件路径
            output_path: 输出文件路径
            save_format: 保存格式
            
        Returns:
            处理结果信息
        """
        try:
            # 加载音频
            signal, sr = self.load_audio(input_path)
            
            # 生成耳蜗电图
            cochleagram = self.generate_cochleagram(signal, sr)
            
            # 保存结果
            self.save_cochleagram(cochleagram, output_path, save_format)
            
            result = {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'signal_shape': signal.shape,
                'cochleagram_shape': cochleagram.shape,
                'sr': sr,
                'duration': len(signal) / sr
            }
            
            self.logger.info(f"处理成功: {input_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"处理失败 {input_path}: {str(e)}")
            return {
                'success': False,
                'input_path': input_path,
                'error': str(e)
            }
    
    def process_batch(self, input_dir: str, output_dir: str, 
                     file_pattern: str = '*.wav', save_format: str = 'npy') -> Dict[str, Any]:
        """
        批量处理音频文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            file_pattern: 文件匹配模式
            save_format: 保存格式
            
        Returns:
            批量处理结果
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # 查找所有匹配的音频文件
        audio_files = list(input_path.glob(file_pattern))
        
        if not audio_files:
            self.logger.warning(f"在 {input_dir} 中没有找到匹配 {file_pattern} 的文件")
            return {'total': 0, 'success': 0, 'failed': 0, 'results': []}
        
        self.logger.info(f"找到 {len(audio_files)} 个音频文件")
        
        results = []
        success_count = 0
        failed_count = 0
        
        for audio_file in audio_files:
            # 构建输出文件路径
            relative_path = audio_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix(f'.{save_format}')
            
            # 处理文件
            result = self.process_single_file(str(audio_file), str(output_file), save_format)
            results.append(result)
            
            if result['success']:
                success_count += 1
            else:
                failed_count += 1
        
        batch_result = {
            'total': len(audio_files),
            'success': success_count,
            'failed': failed_count,
            'results': results
        }
        
        self.logger.info(f"批量处理完成: 总计 {len(audio_files)}, 成功 {success_count}, 失败 {failed_count}")
        return batch_result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='音频文件转耳蜗电图预处理工具')
    parser.add_argument('input', help='输入音频文件或目录路径')
    parser.add_argument('output', help='输出文件或目录路径')
    parser.add_argument('--sr', type=int, default=16000, help='采样率 (默认: 16000)')
    parser.add_argument('--n-filters', type=int, default=64, help='滤波器数量 (默认: 64)')
    parser.add_argument('--low-lim', type=int, default=50, help='低频限制 (默认: 50Hz)')
    parser.add_argument('--hi-lim', type=int, default=20000, help='高频限制 (默认: 20000Hz)')
    parser.add_argument('--sample-factor', type=int, default=1, choices=[1, 2, 4], 
                       help='采样因子 (默认: 1)')
    parser.add_argument('--downsample', type=int, help='下采样因子 (默认: 不下采样)')
    parser.add_argument('--nonlinearity', choices=['db', 'power', 'none'], default='power',
                       help='非线性变换类型 (默认: power)')
    parser.add_argument('--max-duration', type=float, default=10.0,
                       help='最大音频时长 (秒) (默认: 10.0)')
    parser.add_argument('--ihc-lowpass-cutoff', type=float, default=3000.0,
                       help='IHC低通滤波器截止频率 (Hz) (默认: 3000.0)')
    parser.add_argument('--ihc-lowpass-order', type=int, default=7,
                       help='IHC低通滤波器阶数 (默认: 7)')
    parser.add_argument('--format', choices=['npy', 'npz', 'png'], default='npy',
                       help='输出格式 (默认: npy)')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    parser.add_argument('--pattern', default='*.wav', help='文件匹配模式 (默认: *.wav)')
    
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = CochleagramPreprocessor(
        sr=args.sr,
        n_filters=args.n_filters,
        low_lim=args.low_lim,
        hi_lim=args.hi_lim,
        sample_factor=args.sample_factor,
        downsample_factor=args.downsample,
        nonlinearity=args.nonlinearity if args.nonlinearity != 'none' else None,
        max_duration=args.max_duration,
        ihc_lowpass_cutoff=args.ihc_lowpass_cutoff,
        ihc_lowpass_order=args.ihc_lowpass_order
    )
    
    if args.batch:
        # 批量处理模式
        result = preprocessor.process_batch(args.input, args.output, args.pattern, args.format)
        print(f"\n批量处理结果:")
        print(f"总计: {result['total']}")
        print(f"成功: {result['success']}")
        print(f"失败: {result['failed']}")
        
        # 显示失败的文件
        failed_files = [r for r in result['results'] if not r['success']]
        if failed_files:
            print(f"\n失败的文件:")
            for f in failed_files:
                print(f"  {f['input_path']}: {f['error']}")
    else:
        # 单文件处理模式
        result = preprocessor.process_single_file(args.input, args.output, args.format)
        if result['success']:
            print(f"\n处理成功:")
            print(f"输入文件: {result['input_path']}")
            print(f"输出文件: {result['output_path']}")
            print(f"信号形状: {result['signal_shape']}")
            print(f"耳蜗电图形状: {result['cochleagram_shape']}")
            print(f"采样率: {result['sr']}Hz")
            print(f"时长: {result['duration']:.2f}秒")
        else:
            print(f"\n处理失败: {result['error']}")


if __name__ == '__main__':
    main() 