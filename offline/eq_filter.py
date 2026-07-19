from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt

def apply_eq(audio_data, sr, band, gain_db):
    """
    اعمال EQ به یک باند فرکانسی خاص
    
    Parameters:
    -----------
    audio_data : np.ndarray
        سیگنال صوتی
    sr : int
        Sample rate
    band : str
        'bass' (20-250 Hz)
        'mid' (250-4000 Hz)
        'treble' (4000-20000 Hz)
    gain_db : float
        میزان تقویت/تضعیف به دسی‌بل
    """
    
    # تعریف محدوده فرکانسی هر باند
    bands = {
        'bass': (20, 250),
        'mid': (250, 4000),
        'treble': (4000, min(20000, sr // 2 - 100))
    }
    
    if band not in bands:
        raise ValueError(f"Invalid band: {band}")
    
    low_freq, high_freq = bands[band]
    
    # طراحی فیلتر Bandpass
    sos = butter(4, [low_freq, high_freq], btype='band', fs=sr, output='sos')
    
    # اعمال فیلتر
    filtered = sosfilt(sos, audio_data)
    
    # محاسبه ضریب تقویت
    gain_linear = 10 ** (gain_db / 20)
    
    # ترکیب سیگنال
    output = audio_data + filtered * (gain_linear - 1)
    
    # نرمال‌سازی
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val
    
    return output


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
    
    # تست‌های افراطی‌تر
    test_cases = [
        ('bass', +15, "bass_extreme.wav"),       # بیس خیلی قوی
        ('treble', +15, "treble_extreme.wav"),   # صدای زیر خیلی تیز
        ('mid', -15, "mid_kill.wav"),            # صدای وسط تقریباً حذف شده (مثل تلفن)
        ('bass', -12, "bass_cut.wav"),           # بیس حذف شده (صدای نازک)
    ]
    
    for band, gain_db, filename in test_cases:
        print(f"Processing: {band} ({gain_db:+d} dB)...")
        eq_audio = apply_eq(audio, sr, band, gain_db)
        output_file = output_dir / filename
        sf.write(str(output_file), eq_audio, sr)
        print(f"Saved: {filename}\n")
    
    print("Done! Now listen to the files:")
    print("  - bass_extreme.wav  → Should sound BOOMY (lots of low frequencies)")
    print("  - treble_extreme.wav → Should sound SHARP/TINNY (lots of high frequencies)")
    print("  - mid_kill.wav      → Should sound like OLD TELEPHONE")
    print("  - bass_cut.wav      → Should sound THIN (no bass)")
