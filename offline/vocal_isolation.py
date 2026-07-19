from pathlib import Path
import subprocess
import shutil

def isolate_vocals(input_file, output_dir):
    """
    جداسازی صدای خواننده از بیت با استفاده از Demucs
    
    Parameters:
    -----------
    input_file : Path
        مسیر فایل ورودی
    output_dir : Path
        مسیر پوشه خروجی
    
    Returns:
    --------
    vocals_file : Path
        مسیر فایل صدای خواننده
    no_vocals_file : Path
        مسیر فایل بدون صدای خواننده (فقط بیت)
    """
    # اجرای demucs
    print("Separating vocals from accompaniment...")
    print("(First run will download the model, ~300MB)")
    
    # demucs خروجی رو در پوشه جداگانه می‌سازه
    temp_output = output_dir / "demucs_temp"
    
    subprocess.run([
        "demucs",
        "--two-stems", "vocals",  # فقط vocals و no_vocals
        "-o", str(temp_output),
        str(input_file)
    ], check=True)
    
    # مسیرهای فایل‌های خروجی (demucs ساختار خاصی داره)
    model_output = temp_output / "htdemucs" / input_file.stem
    vocals_source = model_output / "vocals.wav"
    no_vocals_source = model_output / "no_vocals.wav"
    
    # انتقال فایل‌ها به پوشه اصلی output
    vocals_file = output_dir / f"{input_file.stem}_vocals.wav"
    no_vocals_file = output_dir / f"{input_file.stem}_no_vocals.wav"
    
    shutil.move(str(vocals_source), str(vocals_file))
    shutil.move(str(no_vocals_source), str(no_vocals_file))
    
    # پاک کردن پوشه موقت
    shutil.rmtree(temp_output)
    
    return vocals_file, no_vocals_file


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    samples_dir = project_root / "samples"
    output_dir = project_root / "output"
    
    input_file = samples_dir / "input.wav"
    
    if not input_file.exists():
        print(f"File not found: {input_file}")
        exit(1)
    
    print(f"Input: {input_file.name}\n")
    
    # جداسازی
    vocals_file, no_vocals_file = isolate_vocals(input_file, output_dir)
    
    print(f"\nDone!")
    print(f"  Vocals only:     {vocals_file.name}")
    print(f"  Instrumental:    {no_vocals_file.name}")
    print("\nListen to both files to hear the separation!")