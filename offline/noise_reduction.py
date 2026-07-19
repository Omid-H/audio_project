from pathlib import Path
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np

def reduce_noise(audio_data, sr, stationary=True, prop_decrease=1.0):
    """
    حذف نویز پس‌زمینه از سیگنال صوتی
    
    Parameters:
    -----------
    audio_data : np.ndarray
        سیگنال صوتی
    sr : int
        نرخ نمونه‌برداری
    stationary : bool
        آیا نویز ثابت است؟ (مثل هیس، فن)
        True → نویز ثابت (بهتر برای هیس)
        False → نویز متغیر (بهتر برای صداهای ناخواسته)
    prop_decrease : float
        میزان کاهش نویز (0.0 تا 1.0)
        1.0 → حذف کامل
        0.5 → حذف نصفی
    """
    reduced = nr.reduce_noise(
        y=audio_data,
        sr=sr,
        stationary=stationary,
        prop_decrease=prop_decrease
    )
    return reduced


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
        (True, 1.0, "stationary_full.wav"),      # حذف کامل نویز ثابت
        (True, 0.5, "stationary_half.wav"),      # حذف نصفی نویز ثابت
        (False, 1.0, "nonstationary_full.wav"),  # حذف کامل نویز متغیر
        (False, 0.5, "nonstationary_half.wav"),  # حذف نصفی نویز متغیر
    ]
    
    for stationary, prop_decrease, filename in test_cases:
        mode = "stationary" if stationary else "non-stationary"
        print(f"Processing: {mode} noise, {int(prop_decrease*100)}% reduction...")
        reduced = reduce_noise(audio, sr, stationary, prop_decrease)
        output_file = output_dir / filename
        sf.write(str(output_file), reduced, sr)
        print(f"Saved: {filename}\n")
    
    print("Done! Listen to the files:")
    print("  - stationary_full.wav      → Full reduction (stationary noise like hiss)")
    print("  - stationary_half.wav      → 50% reduction (stationary)")
    print("  - nonstationary_full.wav   → Full reduction (non-stationary noise)")
    print("  - nonstationary_half.wav   → 50% reduction (non-stationary)")