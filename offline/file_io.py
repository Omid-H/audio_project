import librosa
import soundfile as sf
from pathlib import Path

# تعریف مسیرهای پروژه
PROJECT_ROOT = Path(__file__).parent.parent  # پوشه audio_project
SAMPLES_DIR = PROJECT_ROOT / 'samples'
OUTPUT_DIR = PROJECT_ROOT / 'output'

def load_audio(file_path, sr=16000):
    """بارگذاری فایل صوتی"""
    audio_data, sample_rate = librosa.load(file_path, sr=sr)
    print(f"✅ File loaded: {file_path}")
    print(f"   Sample Rate: {sample_rate} Hz")
    print(f"   Duration: {len(audio_data)/sample_rate:.2f} seconds")
    return audio_data, sample_rate

def save_audio(file_path, audio_data, sr=16000):
    """ذخیره فایل صوتی"""
    sf.write(file_path, audio_data, sr)
    print(f"✅ File saved: {file_path}")

if __name__ == "__main__":
    # تست: بارگذاری و ذخیره
    input_file = SAMPLES_DIR / 'input.wav'
    output_file = OUTPUT_DIR / 'output.wav'
    
    # بررسی وجود فایل ورودی
    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        exit(1)
    
    audio, sr = load_audio(str(input_file))
    save_audio(str(output_file), audio, sr)
