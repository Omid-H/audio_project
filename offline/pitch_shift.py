import librosa
import soundfile as sf
from pathlib import Path

# تعریف مسیرهای پروژه
PROJECT_ROOT = Path(__file__).parent.parent
SAMPLES_DIR = PROJECT_ROOT / 'samples'
OUTPUT_DIR = PROJECT_ROOT / 'output'

def pitch_shift(audio_data, sr, n_steps):
    """
    تغییر pitch صدا
    n_steps: تعداد نیم‌پرده (semitone)
        مثبت = زیرتر (higher pitch)
        منفی = بم‌تر (lower pitch)
    """
    shifted = librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=n_steps)
    return shifted

if __name__ == "__main__":
    input_file = SAMPLES_DIR / 'input.wav'
    
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        exit(1)
    
    # بارگذاری فایل
    audio, sr = librosa.load(str(input_file), sr=16000)
    print(f"Loaded: {input_file.name} ({len(audio)/sr:.2f}s)")
    
    # تست چند حالت مختلف
    test_cases = [
        (-5, "lower_pitch.wav"),   # بم‌تر (مرد)
        (0, "original.wav"),        # بدون تغییر
        (+5, "higher_pitch.wav")    # زیرتر (زن)
    ]
    
    for n_steps, filename in test_cases:
        shifted = pitch_shift(audio, sr, n_steps)
        output_file = OUTPUT_DIR / filename
        sf.write(str(output_file), shifted, sr)
        print(f"Saved: {filename} (pitch shift: {n_steps:+d} semitones)")
