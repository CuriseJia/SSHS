#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data preprocessing pipeline based on pycochleagram library
Convert .wav format audio to cochleagram and save
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

import matplotlib
matplotlib.use('Agg')

from pycochleagram import cochleagram as cgram
from pycochleagram import erbfilter as erb
from pycochleagram import utils


class CochleagramPreprocessor:
    """Cochleagram data preprocessor"""
    
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
        Initialize preprocessor
        
        Args:
            sr: Sampling rate
            n_filters: Number of filters (default 64)
            low_lim: Low frequency limit (Hz)
            hi_lim: High frequency limit (Hz)
            sample_factor (1, 2, 4)
            downsample_factor: downsampling factor, None indicates no downsampling
            nonlinearity: Nonlinear transformation type ('db ',' power ', None)
            strict: whether to use strict mode
            max_duration: Maximum audio duration (seconds)
            target_duration: Target audio duration (seconds), used to standardize the temporal dimension of cochlear electrograms
            ihc_lowpass_cutoff: Cut off frequency of IHC low-pass filter (Hz, default 3000Hz)
            ihc_lowpass_order: IHC low-pass filter order (default order 7)
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
        Load audio file
        
        Args:
            audio_path: Audio file path
            
        Returns:
            (signal, sr): Audio signal and sampling rate
        """
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Use pycochleagram's utils to load audio
        signal, sr = utils.wav_to_array(audio_path)
        
        # Keep multi-channel; process left and right channels separately later when generating cochleagram
        if len(signal.shape) > 1 and signal.shape[1] > 1:
            self.logger.info(f"Detected stereo audio, will process left and right channels separately (shape: {signal.shape})")
        
        # If need to resample
        if sr != self.sr:
            self.logger.info(f"Resampling from {sr}Hz to {self.sr}Hz")
            # Here can add resampling logic
            # Currently use simple downsampling
            if sr > self.sr:
                factor = sr // self.sr
                signal = signal[::factor]
                sr = self.sr
        
        # Truncate audio to specified duration (support single/multi-channel)
        max_samples = int(self.max_duration * sr)
        num_samples = signal.shape[0]
        if num_samples > max_samples:
            self.logger.info(f"Truncating audio from {num_samples} samples to {max_samples} samples (duration: {self.max_duration} seconds)")
            signal = signal[:max_samples, ...] if signal.ndim > 1 else signal[:max_samples]
        else:
            self.logger.info(f"Audio duration: {num_samples / sr:.2f} seconds (not exceeding limit {self.max_duration} seconds)")

        # If need to align to fixed 2-5 second three-second window
        if self.align_window_2_to_5:
            need_total = int(5.0 * sr)
            cur_len = signal.shape[0]
            if cur_len < need_total:
                pad = need_total - cur_len
                if signal.ndim == 1:
                    signal = np.pad(signal, (0, pad), mode='constant', constant_values=0)
                else:
                    signal = np.pad(signal, ((0, pad), (0, 0)), mode='constant', constant_values=0)
                self.logger.info(f"Padding 2-5 second window to 5 seconds: pad={pad} samples")
            start = int(2.0 * sr)
            end = int(5.0 * sr)
            signal = signal[start:end, ...] if signal.ndim > 1 else signal[start:end]
            self.logger.info(f"Truncated to fixed window [2s, 5s): {signal.shape[0]} samples")
        else:
            # If specified target duration, perform standardization processing
            if self.target_duration is not None:
                target_samples = int(self.target_duration * sr)
                self.logger.info(f"Target duration: {self.target_duration} seconds, target samples: {target_samples}")
                cur_len = signal.shape[0]
                if cur_len < target_samples:
                    # Pad zero (support multi-channel)
                    padding = target_samples - cur_len
                    if signal.ndim == 1:
                        signal = np.pad(signal, (0, padding), mode='constant', constant_values=0)
                    else:
                        signal = np.pad(signal, ((0, padding), (0, 0)), mode='constant', constant_values=0)
                    self.logger.info(f"Padding audio from {cur_len} samples to {target_samples} samples")
                elif cur_len > target_samples:
                    # Truncate (support multi-channel)
                    signal = signal[:target_samples, ...] if signal.ndim > 1 else signal[:target_samples]
                    self.logger.info(f"Truncating audio from {cur_len} samples to {target_samples} samples")
                else:
                    self.logger.info(f"Audio length matches target length: {signal.shape[0]} samples")
        
        return signal, sr
    
    def generate_cochleagram(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """
        Generate cochleagram
        
        Args:
            signal: Audio signal
            sr: Sampling rate
            
        Returns:
            cochleagram: Cochleagram array
        """
        # Remove cochleagram generation start log
        
        # Dynamic adjustment of high frequency limit, avoid exceeding Nyquist frequency
        nyquist_freq = sr // 2
        adjusted_hi_lim = min(self.hi_lim, nyquist_freq)
        if adjusted_hi_lim != self.hi_lim:
            self.logger.info(f"Adjusting high frequency limit from {self.hi_lim}Hz to {adjusted_hi_lim}Hz (Nyquist frequency: {nyquist_freq}Hz)")
        
        # Record IHC low-pass filter parameters for phase locking control
        # self.logger.info(f"IHC低通滤波器参数: 截止频率={self.ihc_lowpass_cutoff}Hz, 阶数={self.ihc_lowpass_order}")
        
        # Smart adjustment of downsampling factor, ensure integer multiple relationship
        adjusted_downsample = self.downsample_factor
        if adjusted_downsample is not None:
            # For pycochleagram's poly mode, downsample parameter is interpreted as env_sr (downsampled sampling rate)
            # Need to ensure audio_sr/env_sr is an integer
            audio_sr = sr
            env_sr = adjusted_downsample  # Here adjusted_downsample is env_sr
            
            # Check if it is an integer multiple relationship
            if audio_sr % env_sr != 0:
                # Find the closest integer env_sr, ensure audio_sr can be divided
                # Calculate downsampling factor
                downsample_factor = audio_sr // env_sr
                target_env_sr = audio_sr // downsample_factor
                
                if target_env_sr != env_sr:
                    self.logger.info(f"Adjusting downsampling factor from {adjusted_downsample} to {target_env_sr} (audio_sr: {audio_sr}, downsampling factor: {downsample_factor})")
                    adjusted_downsample = target_env_sr
        
        # Generate cochleagram (support single/dual channel). If it is stereo, process left and right channels separately and stack in the last dimension as 2 channels.
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

            # Last dimension is channel number: [F, T, 2]
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
        Save cochleagram
        
        Args:
            cochleagram: Cochleagram array
            output_path: Output path
            save_format: Save format ('npy', 'npz', 'png')
        """
        self.logger.info(f"Saving cochleagram to: {output_path}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create when path contains directory
            os.makedirs(output_dir, exist_ok=True)
        
        if save_format == 'npy':
            np.save(output_path, cochleagram)
        elif save_format == 'npz':
            np.savez_compressed(output_path, cochleagram=cochleagram)
        elif save_format == 'png':
            self._save_as_image(cochleagram, output_path)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")
            
        self.logger.info(f"Cochleagram saved to: {output_path}")
    
    def _apply_bez2018_phase_locking(self, cochleagram: np.ndarray, sr: int) -> np.ndarray:
        """
        Phase locking control based on BEZ2018 model
        
        According to the literature, phase locking is controlled by adjusting the cutoff frequency of the IHC low-pass filter.
        In the unmodified auditory nerve model, the low-pass characteristic of IHC membrane potential is modeled as a 7th order filter with a cutoff frequency of 3000Hz.
        
        Args:
            cochleagram: Cochleagram array
            sr: Sampling rate
            
        Returns:
            Processed cochleagram
        """
        
        # Calculate the center frequency for each filter
        n_filters = cochleagram.shape[0]
        freqs = np.logspace(np.log10(self.low_lim), np.log10(self.hi_lim), n_filters)
        
        # Apply phase locking control based on IHC low-pass filter
        for i, freq in enumerate(freqs):
            # For frequencies higher than the IHC low-pass filter cutoff frequency, apply phase locking limit
            if freq > self.ihc_lowpass_cutoff:
                # Calculate phase locking attenuation factor
                # Based on BEZ2018 model's IHC low-pass filter characteristic
                attenuation_factor = self._calculate_phase_locking_attenuation(freq)
                
                # Apply attenuation
                cochleagram[i, :] = cochleagram[i, :] * attenuation_factor
                
                self.logger.debug(f"Filter {i} (center frequency: {freq:.1f}Hz) applied phase locking attenuation: {attenuation_factor:.3f}")
        
        return cochleagram
    
    def _calculate_phase_locking_attenuation(self, frequency: float) -> float:
        """
        Calculate phase locking attenuation factor based on BEZ2018 model
        
        Args:
            frequency: Frequency (Hz)
            
        Returns:
            Attenuation factor (between 0 and 1)
        """
        # Based on BEZ2018 model's IHC low-pass filter characteristic
        # Using Butterworth low-pass filter's frequency response characteristic
        cutoff_freq = self.ihc_lowpass_cutoff
        order = self.ihc_lowpass_order
        
        # Calculate normalized frequency
        normalized_freq = frequency / cutoff_freq
        
        # Calculate Butterworth low-pass filter's magnitude response
        # |H(f)| = 1 / sqrt(1 + (f/fc)^(2n))
        magnitude_response = 1.0 / np.sqrt(1.0 + (normalized_freq ** (2 * order)))
        
        # For phase locking, we use a stricter attenuation
        # According to the literature, phase locking starts to significantly decrease above 1500Hz
        if frequency > 1500:
            # Additional phase locking attenuation
            phase_lock_attenuation = np.exp(-(frequency - 1500) / 1000)
            magnitude_response *= phase_lock_attenuation
        
        return magnitude_response
    
    def _save_as_image(self, cochleagram: np.ndarray, output_path: str) -> None:
        """Save as image file (support single/dual channel)."""
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
            # Fall back to displaying the first channel, avoid exception
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
        Process single audio file
        
        Args:
            input_path: Input audio file path
            output_path: Output file path
            save_format: Save format
            
        Returns:
            Processing result information
        """
        try:
            # Load audio
            signal, sr = self.load_audio(input_path)
            
            # Generate cochleagram
            cochleagram = self.generate_cochleagram(signal, sr)
            
            # Save result
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
            
            self.logger.info(f"Processing successful: {input_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Processing failed {input_path}: {str(e)}")
            return {
                'success': False,
                'input_path': input_path,
                'error': str(e)
            }
    
    def process_batch(self, input_dir: str, output_dir: str, 
                     file_pattern: str = '*.wav', save_format: str = 'npy') -> Dict[str, Any]:
        """
        Batch process audio files
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            file_pattern: File matching pattern
            save_format: Save format
            
        Returns:
            Batch processing result
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Find all matching audio files
        audio_files = list(input_path.glob(file_pattern))
        
        if not audio_files:
            self.logger.warning(f"No matching files found in {input_dir} for {file_pattern}")
            return {'total': 0, 'success': 0, 'failed': 0, 'results': []}
        
        self.logger.info(f"Found {len(audio_files)} audio files")
        
        results = []
        success_count = 0
        failed_count = 0
        
        for audio_file in audio_files:
            # Build output file path
            relative_path = audio_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix(f'.{save_format}')
            
            # Process file
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
        
        self.logger.info(f"Batch processing completed: Total {len(audio_files)}, Success {success_count}, Failed {failed_count}")
        return batch_result


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Audio file to cochleagram preprocessing tool')
    parser.add_argument('input', help='Input audio file or directory path')
    parser.add_argument('output', help='Output file or directory path')
    parser.add_argument('--sr', type=int, default=16000, help='Sampling rate (default: 16000)')
    parser.add_argument('--n-filters', type=int, default=64, help='Number of filters (default: 64)')
    parser.add_argument('--low-lim', type=int, default=50, help='Low frequency limit (default: 50Hz)')
    parser.add_argument('--hi-lim', type=int, default=20000, help='High frequency limit (default: 20000Hz)')
    parser.add_argument('--sample-factor', type=int, default=1, choices=[1, 2, 4], 
                       help='Sampling factor (default: 1)')
    parser.add_argument('--downsample', type=int, help='Downsampling factor (default: no downsampling)')
    parser.add_argument('--nonlinearity', choices=['db', 'power', 'none'], default='power',
                       help='Nonlinear transformation type (default: power)')
    parser.add_argument('--max-duration', type=float, default=10.0,
                       help='Maximum audio duration (seconds) (default: 10.0)')
    parser.add_argument('--ihc-lowpass-cutoff', type=float, default=3000.0,
                       help='IHC low-pass filter cutoff frequency (Hz) (default: 3000.0)')
    parser.add_argument('--ihc-lowpass-order', type=int, default=7,
                       help='IHC low-pass filter order (default: 7)')
    parser.add_argument('--format', choices=['npy', 'npz', 'png'], default='npy',
                       help='Output format (default: npy)')
    parser.add_argument('--batch', action='store_true', help='Batch processing mode')
    parser.add_argument('--pattern', default='*.wav', help='File matching pattern (default: *.wav)')
    
    args = parser.parse_args()
    
    # Create preprocessor
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
        # Batch processing mode
        result = preprocessor.process_batch(args.input, args.output, args.pattern, args.format)
        print(f"\nBatch processing result:")
        print(f"Total: {result['total']}")
        print(f"Success: {result['success']}")
        print(f"Failed: {result['failed']}")
        
        # Display failed files
        failed_files = [r for r in result['results'] if not r['success']]
        if failed_files:
            print(f"\nFailed files:")
            for f in failed_files:
                print(f"  {f['input_path']}: {f['error']}")
    else:
        # Single file processing mode
        result = preprocessor.process_single_file(args.input, args.output, args.format)
        if result['success']:
            print(f"\nProcessing successful:")
            print(f"Input file: {result['input_path']}")
            print(f"Output file: {result['output_path']}")
            print(f"Signal shape: {result['signal_shape']}")
            print(f"Cochleagram shape: {result['cochleagram_shape']}")
            print(f"Sampling rate: {result['sr']}Hz")
            print(f"Duration: {result['duration']:.2f} seconds")
        else:
            print(f"\nProcessing failed: {result['error']}")


if __name__ == '__main__':
    main() 