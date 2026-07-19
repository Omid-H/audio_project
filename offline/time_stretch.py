from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

def time_stretch(audio_data, rate):
    """
    تغییر سرعت صدا بدون تغییر pitch
    
    Parameters:
    -----------
    audio_data : np.ndarray
        سیگنال صوتی
    rate : float
        ضریب سرعت
        rate < 1.0 → آهسته‌تر
        rate > 1.0 → سریع‌تر
        rate = 1.0 → بدون تغییر
    """
    stretched = librosa.effects.time_stretch(audio_data, rate=rate)
    
    # Normalize: حداکثر مقدار رو به 0.95 می‌رسونیم (برای جلوگیری از clipping)
    max_val = np.abs(stretched).max()
    if max_val > 0:
        stretched = stretched * (0.95 / max_val)
    
    return stretched


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    samples_dir = project_root / "samples"
    output_dir = project_root / "output"
    
    input_file = samples_dir / "input.wav"
    
    if not input_file.exists():
        print(f"File not found: {input_file}")
        exit(1)
    
    # بارگذاری
    audio, sr = librosa.load(str(input_file), sr=16000)
    print(f"Loaded: {input_file.name} ({len(audio)/sr:.2f}s, {sr} Hz)\n")
    
    # تست‌های مختلف
    test_cases = [
        (0.75, "slow_75.wav"),      # ۲۵٪ آهسته‌تر
        (1.0, "normal.wav"),         # بدون تغییر
        (1.25, "fast_125.wav"),      # ۲۵٪ سریع‌تر
        (1.5, "fast_150.wav"),       # ۵۰٪ سریع‌تر
        (0.5, "slow_50.wav"),        # ۵۰٪ آهسته‌تر (خیلی کند)
    ]
    
    for rate, filename in test_cases:
        print(f"Processing: rate={rate}x...")
        stretched = time_stretch(audio, rate)
        output_file = output_dir / filename
        sf.write(str(output_file), stretched, sr)
        duration = len(stretched) / sr
        print(f"Saved: {filename} (duration: {duration:.2f}s)\n")
    
    print("Done! Listen to the files:")
    print("  - slow_50.wav   → Half speed (very slow)")
    print("  - slow_75.wav   → 75% speed")
    print("  - normal.wav    → Original speed")
    print("  - fast_125.wav  → 125% speed")
    print("  - fast_150.wav  → 150% speed (very fast)")
