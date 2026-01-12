#!/usr/bin/env python3
"""
Voice File Analyzer Utility
Extracts acoustic features from audio files including formants, MFCCs, spectral characteristics.
"""

import argparse
import json
import numpy as np
import librosa
import soundfile as sf
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import sys
from enum import Enum
import warnings
import traceback

warnings.filterwarnings('ignore')

class OutputFormat(Enum):
    JSON = 'json'
    SUMMARY = 'summary'
    FULL = 'full'

@dataclass
class SpectralCharacteristics:
    """Spectral characteristics for a frame or entire audio"""
    centroid: Optional[float] = None
    bandwidth: Optional[float] = None
    contrast: Optional[List[float]] = None
    flatness: Optional[float] = None
    rolloff: Optional[float] = None
    zero_crossing_rate: Optional[float] = None
    rms_energy: Optional[float] = None

@dataclass
class FormantData:
    """Formant frequency and bandwidth data"""
    f1: Optional[float] = None
    f2: Optional[float] = None
    f3: Optional[float] = None
    f4: Optional[float] = None
    b1: Optional[float] = None
    b2: Optional[float] = None
    b3: Optional[float] = None
    b4: Optional[float] = None

@dataclass
class FrameData:
    """Detailed frame-by-frame acoustic data"""
    frame_index: int
    timestamp: float
    mfcc: Optional[List[float]] = None
    spectral: Optional[SpectralCharacteristics] = None
    formants: Optional[FormantData] = None
    pitch: Optional[float] = None
    pitch_confidence: Optional[float] = None
    voiced: Optional[bool] = None

@dataclass
class VoiceAnalysisResult:
    """Complete voice analysis results"""
    # File metadata
    filename: str
    duration: float
    sample_rate: int
    channels: int
    total_frames: int

    # Global features
    global_mfcc: Optional[List[float]] = None
    global_spectral: Optional[SpectralCharacteristics] = None
    mean_formants: Optional[FormantData] = None

    # Statistical features
    pitch_statistics: Dict[str, float] = field(default_factory=dict)
    energy_statistics: Dict[str, float] = field(default_factory=dict)

    # Frame-by-frame data (if requested)
    frames: Optional[List[FrameData]] = None

    # Processing parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

class VoiceAnalyzer:
    """Main voice analysis utility"""

    def __init__(self,
                 sample_rate: int = 22050,
                 hop_length: int = 512,
                 n_mfcc: int = 13,
                 n_formants: int = 4,
                 max_f0: float = 500.0,
                 min_f0: float = 80.0):
        """
        Initialize voice analyzer with processing parameters.

        Args:
            sample_rate: Target sample rate for analysis
            hop_length: Hop length between frames
            n_mfcc: Number of MFCC coefficients
            n_formants: Number of formants to track
            max_f0: Maximum fundamental frequency (Hz)
            min_f0: Minimum fundamental frequency (Hz)
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.n_formants = n_formants
        self.max_f0 = max_f0
        self.min_f0 = min_f0

    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to mono if needed.

        Args:
            filepath: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            raise IOError(f"Failed to load audio file {filepath}: {str(e)}")

    def extract_pitch_pyin(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch using pYIN algorithm.

        Args:
            audio: Audio signal

        Returns:
            Tuple of (pitches, confidence_measures)
        """
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fmin=self.min_f0,
            fmax=self.max_f0
        )

        # Extract predominant pitch per frame
        pitch_vals = []
        confidence_vals = []

        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch_val = pitches[index, i]
            confidence = magnitudes[index, i]

            pitch_vals.append(pitch_val if pitch_val > 0 else 0)
            confidence_vals.append(confidence)

        return np.array(pitch_vals), np.array(confidence_vals)

    def extract_formants_lpc(self, audio: np.ndarray, pitch: np.ndarray,
                           order: int = 12) -> List[FormantData]:
        """
        Estimate formants using Linear Predictive Coding (LPC).

        Args:
            audio: Audio signal
            pitch: Pitch values for each frame
            order: LPC order

        Returns:
            List of FormantData objects for each frame
        """
        try:
            import scipy.signal as spsignal
            from scipy.signal import lfilter
        except ImportError:
            print("Warning: scipy not installed. Formant extraction disabled.", file=sys.stderr)
            return []

        formants_list = []
        frame_length = self.hop_length * 2

        for i in range(0, len(audio) - frame_length, self.hop_length):
            frame = audio[i:i + frame_length]

            if len(frame) < order + 1:
                formants_list.append(FormantData())
                continue

            # Apply window
            windowed = frame * np.hamming(len(frame))

            # LPC analysis
            try:
                a = librosa.lpc(windowed, order=order)
                roots = np.roots(a)
                roots = roots[np.imag(roots) >= 0]

                # Calculate formant frequencies
                angles = np.arctan2(np.imag(roots), np.real(roots))
                frequencies = angles * (self.sample_rate / (2 * np.pi))

                # Sort and select formants
                sorted_idx = np.argsort(frequencies)
                frequencies = frequencies[sorted_idx]

                # Filter formants (typically between 200-5000 Hz)
                formant_mask = (frequencies > 200) & (frequencies < 5000)
                frequencies = frequencies[formant_mask]

                # Create FormantData object
                formant_data = FormantData()
                for j in range(min(self.n_formants, len(frequencies))):
                    setattr(formant_data, f'f{j+1}', float(frequencies[j]))

                formants_list.append(formant_data)

            except:
                formants_list.append(FormantData())

        return formants_list

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features.

        Args:
            audio: Audio signal

        Returns:
            MFCC matrix (n_mfcc x n_frames)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length
        )
        return mfcc

    def extract_spectral_features(self, audio: np.ndarray) -> Tuple[SpectralCharacteristics, List[SpectralCharacteristics]]:
        """
        Extract spectral characteristics.

        Args:
            audio: Audio signal

        Returns:
            Tuple of (global_spectral, frame_spectral_list)
        """
        try:
            # Global features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )[0]

            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )[0]

            spectral_contrast = librosa.feature.spectral_contrast(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )

            spectral_flatness = librosa.feature.spectral_flatness(
                y=audio, hop_length=self.hop_length
            )[0]

            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )[0]

            zero_crossing_rate = librosa.feature.zero_crossing_rate(
                y=audio, hop_length=self.hop_length
            )[0]

            rms_energy = librosa.feature.rms(
                y=audio, hop_length=self.hop_length
            )[0]

            # Global spectral characteristics
            global_spectral = SpectralCharacteristics(
                centroid=float(np.mean(spectral_centroid)),
                bandwidth=float(np.mean(spectral_bandwidth)),
                contrast=np.mean(spectral_contrast, axis=1).tolist() if spectral_contrast.size > 0 else None,
                flatness=float(np.mean(spectral_flatness)),
                rolloff=float(np.mean(spectral_rolloff)),
                zero_crossing_rate=float(np.mean(zero_crossing_rate)),
                rms_energy=float(np.mean(rms_energy))
            )

            # Per-frame spectral characteristics
            frame_spectral_list = []
            n_frames = len(spectral_centroid)

            for i in range(n_frames):
                frame_spec = SpectralCharacteristics(
                    centroid=float(spectral_centroid[i]) if i < len(spectral_centroid) else None,
                    bandwidth=float(spectral_bandwidth[i]) if i < len(spectral_bandwidth) else None,
                    contrast=spectral_contrast[:, i].tolist() if i < spectral_contrast.shape[1] else None,
                    flatness=float(spectral_flatness[i]) if i < len(spectral_flatness) else None,
                    rolloff=float(spectral_rolloff[i]) if i < len(spectral_rolloff) else None,
                    zero_crossing_rate=float(zero_crossing_rate[i]) if i < len(zero_crossing_rate) else None,
                    rms_energy=float(rms_energy[i]) if i < len(rms_energy) else None
                )
                frame_spectral_list.append(frame_spec)

            return global_spectral, frame_spectral_list

        except Exception as e:
            print(f"Warning: Error extracting spectral features: {str(e)}", file=sys.stderr)
            # Return empty/default spectral characteristics
            return SpectralCharacteristics(), []

    def calculate_statistics(self, pitch: np.ndarray, frame_spectral: List[SpectralCharacteristics]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate pitch and energy statistics"""
        # Pitch statistics
        voiced_pitches = pitch[pitch > 0]
        pitch_stats = {
            'mean': float(np.mean(voiced_pitches)) if len(voiced_pitches) > 0 else 0.0,
            'std': float(np.std(voiced_pitches)) if len(voiced_pitches) > 0 else 0.0,
            'min': float(np.min(voiced_pitches)) if len(voiced_pitches) > 0 else 0.0,
            'max': float(np.max(voiced_pitches)) if len(voiced_pitches) > 0 else 0.0,
            'median': float(np.median(voiced_pitches)) if len(voiced_pitches) > 0 else 0.0,
            'voicing_rate': len(voiced_pitches) / len(pitch) if len(pitch) > 0 else 0.0
        }

        # Energy statistics
        energy_values = []
        for fs in frame_spectral:
            if fs.rms_energy is not None:
                energy_values.append(fs.rms_energy)

        if energy_values:
            energy_stats = {
                'mean': float(np.mean(energy_values)),
                'std': float(np.std(energy_values)) if len(energy_values) > 1 else 0.0,
                'min': float(np.min(energy_values)),
                'max': float(np.max(energy_values)),
                'dynamic_range': float(np.max(energy_values) - np.min(energy_values))
            }
        else:
            energy_stats = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'dynamic_range': 0.0
            }

        return pitch_stats, energy_stats

    def analyze(self, filepath: str, include_frames: bool = False) -> VoiceAnalysisResult:
        """
        Perform comprehensive voice analysis.

        Args:
            filepath: Path to audio file
            include_frames: Whether to include frame-by-frame data

        Returns:
            VoiceAnalysisResult object
        """
        try:
            # Load audio
            print(f"Loading audio file...", file=sys.stderr)
            audio, original_sr = self.load_audio(filepath)
            duration = librosa.get_duration(y=audio, sr=self.sample_rate)

            # Extract features
            print(f"Extracting pitch (pYIN)...", file=sys.stderr)
            pitch, confidence = self.extract_pitch_pyin(audio)

            print(f"Extracting formants (LPC)...", file=sys.stderr)
            formants = self.extract_formants_lpc(audio, pitch)

            print(f"Extracting MFCCs...", file=sys.stderr)
            mfcc = self.extract_mfcc(audio)

            print(f"Extracting spectral characteristics...", file=sys.stderr)
            global_spectral, frame_spectral = self.extract_spectral_features(audio)

            # Calculate statistics
            print(f"Calculating statistics...", file=sys.stderr)
            pitch_stats, energy_stats = self.calculate_statistics(pitch, frame_spectral)

            # Calculate mean formants
            mean_formants = FormantData()
            if formants:
                formant_values = {f'f{i}': [] for i in range(1, self.n_formants + 1)}

                for f in formants:
                    for i in range(1, self.n_formants + 1):
                        attr_name = f'f{i}'
                        val = getattr(f, attr_name)
                        if val is not None:
                            formant_values[attr_name].append(val)

                for i in range(1, self.n_formants + 1):
                    attr_name = f'f{i}'
                    if formant_values[attr_name]:
                        setattr(mean_formants, attr_name, float(np.mean(formant_values[attr_name])))

            # Prepare frame data if requested
            frames = None
            if include_frames:
                print(f"Preparing frame-by-frame data...", file=sys.stderr)
                frames = []
                n_frames = min(len(pitch), len(formants), len(frame_spectral), mfcc.shape[1])

                for i in range(n_frames):
                    frame_mfcc = mfcc[:, i].tolist() if i < mfcc.shape[1] else None

                    frame = FrameData(
                        frame_index=i,
                        timestamp=i * self.hop_length / self.sample_rate,
                        mfcc=frame_mfcc,
                        spectral=frame_spectral[i] if i < len(frame_spectral) else None,
                        formants=formants[i] if i < len(formants) else None,
                        pitch=float(pitch[i]) if i < len(pitch) else None,
                        pitch_confidence=float(confidence[i]) if i < len(confidence) else None,
                        voiced=pitch[i] > 0 if i < len(pitch) else None
                    )
                    frames.append(frame)

            # Prepare result
            print(f"Finalizing results...", file=sys.stderr)
            result = VoiceAnalysisResult(
                filename=Path(filepath).name,
                duration=duration,
                sample_rate=self.sample_rate,
                channels=1,  # Always mono after conversion
                total_frames=mfcc.shape[1],
                global_mfcc=np.mean(mfcc, axis=1).tolist() if mfcc.size > 0 else None,
                global_spectral=global_spectral,
                mean_formants=mean_formants,
                pitch_statistics=pitch_stats,
                energy_statistics=energy_stats,
                frames=frames,
                parameters={
                    'sample_rate': self.sample_rate,
                    'hop_length': self.hop_length,
                    'n_mfcc': self.n_mfcc,
                    'n_formants': self.n_formants,
                    'max_f0': self.max_f0,
                    'min_f0': self.min_f0
                }
            )

            return result

        except Exception as e:
            print(f"Error during analysis: {str(e)}", file=sys.stderr)
            traceback.print_exc()
            raise

class VoiceAnalysisEncoder(json.JSONEncoder):
    """Custom JSON encoder for voice analysis objects"""
    def default(self, obj):
        if isinstance(obj, (VoiceAnalysisResult, FrameData, SpectralCharacteristics, FormantData)):
            return asdict(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

def save_results(result: VoiceAnalysisResult, output_file: str, format: OutputFormat):
    """Save analysis results to file"""
    try:
        if format == OutputFormat.JSON or format == OutputFormat.FULL:
            if output_file is None:
                json.dump(result, sys.stdout, cls=VoiceAnalysisEncoder, indent=2)
            else:
                with open(output_file, 'w') as f:
                    json.dump(result, f, cls=VoiceAnalysisEncoder, indent=2)
        elif format == OutputFormat.SUMMARY:
            summary = {
                'filename': result.filename,
                'duration': result.duration,
                'sample_rate': result.sample_rate,
                'pitch_statistics': result.pitch_statistics,
                'energy_statistics': result.energy_statistics,
                'mean_formants': asdict(result.mean_formants) if result.mean_formants else {},
                'mfcc_mean': result.global_mfcc,
                'spectral_centroid': result.global_spectral.centroid if result.global_spectral else None
            }
            if output_file is None:
                json.dump(summary, sys.stdout, indent=2)
            else:
                with open(output_file, 'w') as f:
                    json.dump(summary, f, indent=2)
        print(f"Results saved to {output_file}", file=sys.stderr)
    except Exception as e:
        print(f"Error saving results: {str(e)}", file=sys.stderr)
        raise

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Voice File Analyzer - Extract acoustic features from audio files"
    )
    parser.add_argument("input_file", help="Input audio file path")
    parser.add_argument("-o", "--output", help="Output file path (default: input_file.json)")
    parser.add_argument("--include-frames", action="store_true",
                       help="Include detailed frame-by-frame data")
    parser.add_argument("--format", choices=['json', 'summary', 'full'],
                       default='json', help="Output format")
    parser.add_argument("--sample-rate", type=int, default=22050,
                       help="Target sample rate for analysis")
    parser.add_argument("--hop-length", type=int, default=512,
                       help="Hop length between frames")
    parser.add_argument("--n-mfcc", type=int, default=13,
                       help="Number of MFCC coefficients")
    parser.add_argument("--n-formants", type=int, default=4,
                       help="Number of formants to track")
    parser.add_argument("--max-f0", type=float, default=500.0,
                       help="Maximum fundamental frequency (Hz)")
    parser.add_argument("--min-f0", type=float, default=80.0,
                       help="Minimum fundamental frequency (Hz)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    # Set output file
    output_file = args.output

    try:
        # Initialize analyzer
        analyzer = VoiceAnalyzer(
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            n_mfcc=args.n_mfcc,
            n_formants=args.n_formants,
            max_f0=args.max_f0,
            min_f0=args.min_f0
        )

        if args.verbose:
            print(f"Analyzing {args.input_file}...", file=sys.stderr)
            print(f"Parameters:", file=sys.stderr)
            print(f"  Sample rate: {args.sample_rate} Hz", file=sys.stderr)
            print(f"  Hop length: {args.hop_length} samples", file=sys.stderr)
            print(f"  MFCC coefficients: {args.n_mfcc}", file=sys.stderr)
            print(f"  Formants: {args.n_formants}", file=sys.stderr)
            print(f"  F0 range: {args.min_f0}-{args.max_f0} Hz", file=sys.stderr)

        # Perform analysis
        result = analyzer.analyze(args.input_file, include_frames=args.include_frames)

        # Save results
        save_results(result, output_file, OutputFormat(args.format))

        if args.verbose:
            print(f"\nAnalysis complete:", file=sys.stderr)
            print(f"  Duration: {result.duration:.2f}s", file=sys.stderr)
            print(f"  Frames: {result.total_frames}", file=sys.stderr)
            print(f"  Mean pitch: {result.pitch_statistics['mean']:.1f} Hz", file=sys.stderr)
            print(f"  Voicing rate: {result.pitch_statistics['voicing_rate']:.2%}", file=sys.stderr)
            if result.mean_formants:
                for i in range(1, args.n_formants + 1):
                    attr = f'f{i}'
                    val = getattr(result.mean_formants, attr)
                    if val:
                        print(f"  Mean {attr.upper()}: {val:.1f} Hz", file=sys.stderr)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
