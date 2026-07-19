import gradio as gr
import soundfile as sf
import numpy as np
import glob
import librosa
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
from offline.pitch_shift import pitch_shift
from offline.eq_filter import apply_eq
from offline.noise_reduction import reduce_noise
from offline.time_stretch import time_stretch
from rvc_python.infer import RVCInference
from pydub import AudioSegment

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR   = PROJECT_ROOT / "output"
MODELS_DIR   = PROJECT_ROOT / "rvc_models"
OUTPUT_DIR.mkdir(exist_ok=True)

rvc = RVCInference(device="cpu")
current_model = {"name": None}

def get_available_models():
    models = {}
    for pth in glob.glob(str(MODELS_DIR / "**" / "*.pth"), recursive=True):
        p = Path(pth)
        if p.stem in ("hubert_base", "rmvpe"):
            continue
        index_candidates = glob.glob(str(p.parent / "*.index"))
        index = index_candidates[0] if index_candidates else ""
        models[p.stem] = {"pth": pth, "index": index}
    return models

def make_mp3(wav_path):
    """از WAV یک نسخه MP3 میسازه"""
    if wav_path is None:
        return None

    mp3_path = wav_path.replace('.wav', '.mp3')

    audio = AudioSegment.from_wav(wav_path)

    # sample rate استاندارد
    audio = audio.set_frame_rate(44100)

    # استریو اختیاری ولی بعضی encoder ها دوست دارند
    audio = audio.set_channels(2)

    audio.export(
        mp3_path,
        format='mp3',
    )

    return mp3_path

def load_rvc_model(model_name):
    models = get_available_models()
    if model_name not in models:
        return f"❌ مدل {model_name} پیدا نشد."
    if current_model["name"] == model_name:
        return f"✅ مدل {model_name} قبلاً بارگذاری شده."
    info = models[model_name]
    rvc.load_model(info["pth"], version="v2", index_path="")
    current_model["name"] = model_name
    return f"✅ مدل {model_name} بارگذاری شد."


def convert_to_wav_if_needed(filepath):
    """اگه فایل MP3 بود، تبدیل به WAV میکنه"""
    if filepath is None:
        return None
    if filepath.lower().endswith('.mp3'):
        audio = AudioSegment.from_mp3(filepath)
        tmp = tempfile.mktemp(suffix='.wav')
        audio.export(tmp, format='wav')
        return tmp
    return filepath

def prepare_audio(mic, upload):
    audio = mic if mic is not None else upload
    if audio is None:
        return None, None, "❌ هیچ صدایی وارد نشده."
    sr, data = audio
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
        if np.abs(data).max() > 1.0:
            data = data / np.abs(data).max()
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr, None

def make_waveform(data, sr, title="Waveform"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 4))
    fig.patch.set_facecolor('#1e1e2e')
    times = np.linspace(0, len(data)/sr, len(data))
    axes[0].plot(times, data, color='#89b4fa', linewidth=0.5)
    axes[0].set_facecolor('#1e1e2e')
    axes[0].set_ylabel("Amplitude", color='white')
    axes[0].tick_params(colors='white')
    axes[0].set_title(title, color='white')
    for spine in axes[0].spines.values():
        spine.set_edgecolor('#444')
    fft = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), 1/sr)
    axes[1].plot(freqs, 20*np.log10(fft + 1e-10), color='#a6e3a1', linewidth=0.5)
    axes[1].set_facecolor('#1e1e2e')
    axes[1].set_xlim(0, sr//2)
    axes[1].set_xlabel("Frequency (Hz)", color='white')
    axes[1].set_ylabel("dB", color='white')
    axes[1].tick_params(colors='white')
    for spine in axes[1].spines.values():
        spine.set_edgecolor('#444')
    plt.tight_layout()
    return fig


def process_voice_conversion(mic, upload, model_name):
    if model_name == "لطفاً یک مدل انتخاب کنید":
        return None, None, None, None, None, "❌ لطفاً ابتدا یک مدل انتخاب کنید."
    data, sr, err = prepare_audio(mic, upload)
    if err:
        return None, None, None, None, None, err
    fig_before = make_waveform(data, sr, "Before Conversion")
    status = load_rvc_model(model_name)
    if "❌" in status:
        return None, None, None, None, None, status
    input_path  = str(OUTPUT_DIR / "vc_input_temp.wav")
    output_path = str(OUTPUT_DIR / "vc_output.wav")
    data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    sr = 16000
    sf.write(input_path, data, sr)
    rvc.set_params(f0method="pm", f0up_key=0)
    rvc.infer_file(input_path, output_path)
    out_data, out_sr = sf.read(output_path)
    fig_after = make_waveform(out_data, out_sr, "After Conversion")
    mp3_path = make_mp3(output_path)
    return output_path, output_path, mp3_path, fig_before, fig_after, f"✅ تبدیل با مدل {model_name} انجام شد."

def process_pitch_shift(mic, upload, n_steps):
    data, sr, err = prepare_audio(mic, upload)
    if err:
        return None, None, None, None, None, err
    fig_before = make_waveform(data, sr, "Before Processing")
    result = pitch_shift(data, sr, int(n_steps))
    output_path = str(OUTPUT_DIR / "pitch_output.wav")
    sf.write(output_path, result, sr)
    fig_after = make_waveform(result, sr, "After Processing")
    mp3_path = make_mp3(output_path)
    return output_path, output_path, mp3_path, fig_before, fig_after, f"✅ Pitch {int(n_steps):+d} semitone تغییر کرد."

def process_eq(mic, upload, band, gain_db):
    data, sr, err = prepare_audio(mic, upload)
    if err:
        return None, None, None, None, None, err
    fig_before = make_waveform(data, sr, "Before EQ Processing")
    result = apply_eq(data, sr, band, float(gain_db))
    output_path = str(OUTPUT_DIR / "eq_output.wav")
    sf.write(output_path, result, sr)
    fig_after = make_waveform(result, sr, "After EQ Processing")
    mp3_path = make_mp3(output_path)
    return output_path, output_path, mp3_path, fig_before, fig_after, f"✅ EQ اعمال شد: {band} {float(gain_db):+.1f} dB"

def process_noise_reduction(mic, upload, prop_decrease):
    data, sr, err = prepare_audio(mic, upload)
    if err:
        return None, None, None, None, None, err
    fig_before = make_waveform(data, sr, "Before Noise Reduction")
    result = reduce_noise(data, sr, stationary=True, prop_decrease=float(prop_decrease))
    output_path = str(OUTPUT_DIR / "noise_output.wav")
    sf.write(output_path, result, sr)
    fig_after = make_waveform(result, sr, "After Noise Reduction")
    mp3_path = make_mp3(output_path)
    return output_path, output_path, mp3_path, fig_before, fig_after, f"✅ نویز {int(float(prop_decrease)*100)}% کاهش یافت."

def process_time_stretch(mic, upload, rate):
    data, sr, err = prepare_audio(mic, upload)
    if err:
        return None, None, None, None, None, err
    fig_before = make_waveform(data, sr, "Before Time Stretch")
    result = time_stretch(data, float(rate))
    output_path = str(OUTPUT_DIR / "stretch_output.wav")
    sf.write(output_path, result, sr)
    fig_after = make_waveform(result, sr, "After Time Stretch")
    mp3_path = make_mp3(output_path)
    return output_path, output_path, mp3_path, fig_before, fig_after, f"✅ سرعت به {float(rate):.2f}x تغییر کرد."

def process_pipeline(mic, upload, nr_en, nr_amt, ps_en, ps_amt, eq_en, eq_band, eq_gain, ts_en, ts_amt):
    from offline.pipeline import run_pipeline
    data, sr, err = prepare_audio(mic, upload)
    if err:
        return None, None, None, None, None, err
    
    steps = []
    if nr_en:
        steps.append({"type": "noise_reduction", "prop_decrease": nr_amt})
    if ps_en:
        steps.append({"type": "pitch_shift", "n_steps": int(ps_amt)})
    if eq_en:
        steps.append({"type": "eq", "band": eq_band, "gain_db": float(eq_gain)})
    if ts_en:
        steps.append({"type": "time_stretch", "rate": float(ts_amt)})
    
    if not steps:
        return None, None, None, None, None, "❌ هیچ effect ای انتخاب نشده."
    fig_before = make_waveform(data, sr, "Before Processing")
    result = run_pipeline(data, sr, steps)
    output_path = str(OUTPUT_DIR / "pipeline_output.wav")
    sf.write(output_path, result, sr)
    fig_after = make_waveform(result, sr, "After Processing")
    mp3_path = make_mp3(output_path)
    
    step_names = []
    if nr_en: step_names.append("Noise Reduction")
    if ps_en: step_names.append("Pitch Shift")
    if eq_en: step_names.append("EQ")
    if ts_en: step_names.append("Time Stretch")
    
    return output_path, output_path, mp3_path, fig_before, fig_after, f"✅ Pipeline اجرا شد: {' → '.join(step_names)}"

def process_vocal_isolation(upload):
    if upload is None:
        return None, None, None, None, "❌ فایل موزیک وارد نشده."
    from offline.vocal_isolation import isolate_vocals
    input_path = Path(upload)
    try:
        vocals_file, no_vocals_file = isolate_vocals(input_path, OUTPUT_DIR)
        return (
            str(vocals_file),
            str(no_vocals_file),
            str(vocals_file),
            str(no_vocals_file),
            "✅ جداسازی انجام شد."
        )
    except Exception as e:
        return None, None, None, None, f"❌ خطا: {str(e)}"

model_list = list(get_available_models().keys())

css = """
body, .gradio-container { background-color: #0f0f1a !important; color: #cdd6f4 !important; }
.tabs { background: #0f0f1a !important; }
.tab-nav button { background: #1e1e2e !important; color: #cdd6f4 !important; border: 1px solid #313244 !important; border-radius: 8px !important; margin: 2px 2px 12px 2px !important; }
.tab-nav { margin-bottom: 16px !important; padding-bottom: 8px !important; }
.tab-nav button.selected { background: #89b4fa !important; color: #1e1e2e !important; font-weight: bold !important; }
.block { background: #1e1e2e !important; border: 1px solid #313244 !important; border-radius: 12px !important; padding: 12px !important; }
label, .label-wrap span { color: #cdd6f4 !important; font-weight: 500 !important; }
input, textarea, select { background: #181825 !important; color: #cdd6f4 !important; border: 1px solid #313244 !important; border-radius: 8px !important; }
button.primary { background: linear-gradient(135deg, #89b4fa, #b4befe) !important; color: #1e1e2e !important; font-weight: bold !important; border: none !important; border-radius: 8px !important; }
button.primary:hover { background: linear-gradient(135deg, #b4befe, #cba6f7) !important; }
footer { display: none !important; }

.gradio-dropdown { min-width: 0 !important; width: 100% !important; }
.gradio-dropdown .wrap { overflow: hidden !important; }

select { max-width: 100% !important; width: 100% !important; overflow: hidden !important; text-overflow: ellipsis !important; }
.gradio-dropdown { max-width: 100% !important; }

h3 { text-align: center !important; direction: rtl !important; }
"""

with gr.Blocks(title="Audio Processing System", theme=gr.themes.Base(), css=css) as app:
    gr.Markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <h1 style='color:#89b4fa; font-size:2em; margin:0;'>🎙️ سیستم پردازش صوت</h1>
        <p style='color:#6c7086; margin-top:6px;'>Real-time Audio Processing System</p>
    </div>
    """)

    with gr.Tab("🔊 Voice Conversion"):
        gr.Markdown("### تبدیل صدا به صدای شخص دیگر")
        with gr.Row():
            with gr.Column():
                vc_mic    = gr.Audio(source="microphone", type="numpy", label="ضبط از میکروفون")
                vc_upload = gr.Audio(source="upload", type="numpy", label="یا آپلود فایل WAV/MP3")
                vc_model = gr.Dropdown( choices=["لطفاً یک مدل انتخاب کنید"] + model_list, value="لطفاً یک مدل انتخاب کنید", label="انتخاب مدل")
                vc_btn    = gr.Button("▶ تبدیل", variant="primary")
            with gr.Column():
                vc_output      = gr.Audio(label="خروجی")
                vc_download_wav = gr.File(label="دانلود WAV")
                vc_download_mp3 = gr.File(label="دانلود MP3")
                vc_wave_before   = gr.Plot(label="نمودار قبل از پردازش")
                vc_wave_after    = gr.Plot(label="نمودار بعد از پردازش")
                vc_status      = gr.Textbox(label="وضعیت", interactive=False)
        vc_btn.click(process_voice_conversion,
                    inputs=[vc_mic, vc_upload, vc_model],
                    outputs=[vc_output, vc_download_wav, vc_download_mp3, vc_wave_before, vc_wave_after, vc_status])

    with gr.Tab("🎵 Pitch Shift"):
        gr.Markdown("### تغییر زیر و بمی صدا")
        with gr.Row():
            with gr.Column():
                ps_mic    = gr.Audio(source="microphone", type="numpy", label="ضبط از میکروفون")
                ps_upload = gr.Audio(source="upload", type="numpy", label="یا آپلود فایل WAV")
                ps_steps  = gr.Slider(-12, 12, value=0, step=1, label="semitone (منفی=بم‌تر، مثبت=زیرتر)")
                ps_btn    = gr.Button("▶ اعمال", variant="primary")
            with gr.Column():
                ps_output        = gr.Audio(label="خروجی")
                ps_download_wav  = gr.File(label="دانلود WAV")
                ps_download_mp3  = gr.File(label="دانلود MP3")
                ps_wave_before   = gr.Plot(label="نمودار قبل از پردازش")
                ps_wave_after    = gr.Plot(label="نمودار بعد از پردازش")
                ps_status        = gr.Textbox(label="وضعیت", interactive=False)
        ps_btn.click(process_pitch_shift, 
                        inputs=[ps_mic, ps_upload, ps_steps],
                        outputs=[ps_output, ps_download_wav, ps_download_mp3, ps_wave_before, ps_wave_after, ps_status])

    with gr.Tab("🎛️ EQ Filter"):
        gr.Markdown("### تنظیم فرکانس‌های صدا")
        with gr.Row():
            with gr.Column():
                eq_mic    = gr.Audio(source="microphone", type="numpy", label="ضبط از میکروفون")
                eq_upload = gr.Audio(source="upload", type="numpy", label="یا آپلود فایل WAV")
                eq_band = gr.Dropdown(["bass", "mid", "treble"], value="bass", label="باند فرکانسی")
                eq_gain   = gr.Slider(-15, 15, value=0, step=1, label="تقویت/تضعیف (dB)")
                eq_btn    = gr.Button("▶ اعمال", variant="primary")
            with gr.Column():
                eq_output   = gr.Audio(label="خروجی")
                eq_download_wav  = gr.File(label="دانلود WAV")
                eq_download_mp3  = gr.File(label="دانلود MP3")
                eq_wave_before   = gr.Plot(label="نمودار قبل از پردازش")
                eq_wave_after    = gr.Plot(label="نمودار بعد از پردازش")
                eq_status   = gr.Textbox(label="وضعیت", interactive=False)
        eq_btn.click(process_eq, 
                    inputs=[eq_mic, eq_upload, eq_band, eq_gain], 
                    outputs=[eq_output, eq_download_wav,eq_download_mp3, eq_wave_before, eq_wave_after, eq_status])

    with gr.Tab("🔇 Noise Reduction"):
        gr.Markdown("### حذف نویز پس‌زمینه")
        with gr.Row():
            with gr.Column():
                nr_mic    = gr.Audio(source="microphone", type="numpy", label="ضبط از میکروفون")
                nr_upload = gr.Audio(source="upload", type="numpy", label="یا آپلود فایل WAV")
                nr_prop   = gr.Slider(0.1, 1.0, value=0.8, step=0.1, label="میزان کاهش نویز")
                nr_btn    = gr.Button("▶ اعمال", variant="primary")
            with gr.Column():
                nr_output   = gr.Audio(label="خروجی")
                nr_download_wav  = gr.File(label="دانلود WAV")
                nr_download_mp3  = gr.File(label="دانلود MP3")
                nr_wave_before   = gr.Plot(label="نمودار قبل از پردازش")
                nr_wave_after    = gr.Plot(label="نمودار بعد از پردازش")
                nr_status   = gr.Textbox(label="وضعیت", interactive=False)
        nr_btn.click(process_noise_reduction, 
                    inputs=[nr_mic, nr_upload, nr_prop], 
                    outputs=[nr_output, nr_download_wav,nr_download_mp3, nr_wave_before, nr_wave_after, nr_status])

    with gr.Tab("⏩ Time Stretch"):
        gr.Markdown("### تغییر سرعت صدا بدون تغییر pitch")
        with gr.Row():
            with gr.Column():
                ts_mic    = gr.Audio(source="microphone", type="numpy", label="ضبط از میکروفون")
                ts_upload = gr.Audio(source="upload", type="numpy", label="یا آپلود فایل WAV")
                ts_rate   = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="ضریب سرعت (1.0=نرمال)")
                ts_btn    = gr.Button("▶ اعمال", variant="primary")
            with gr.Column():
                ts_output   = gr.Audio(label="خروجی")
                ts_download_wav  = gr.File(label="دانلود WAV")
                ts_download_mp3  = gr.File(label="دانلود MP3")
                ts_wave_before   = gr.Plot(label="نمودار قبل از پردازش")
                ts_wave_after    = gr.Plot(label="نمودار بعد از پردازش")
                ts_status   = gr.Textbox(label="وضعیت", interactive=False)
        ts_btn.click(process_time_stretch, 
                    inputs=[ts_mic, ts_upload, ts_rate], 
                    outputs=[ts_output, ts_download_wav, ts_download_mp3, ts_wave_before, ts_wave_after, ts_status])

    with gr.Tab("⚙️ Pipeline"):
        gr.Markdown("### ترکیب چند effect روی یک صدا")
        with gr.Row():
            with gr.Column():
                pl_mic    = gr.Audio(source="microphone", type="numpy", label="ضبط از میکروفون")
                pl_upload = gr.Audio(source="upload", type="numpy", label="یا آپلود فایل WAV/MP3")
                
                gr.Markdown("#### مراحل پردازش:")
                
                with gr.Group():
                    nr_enable = gr.Checkbox(label="Noise Reduction", value=False)
                    nr_amount = gr.Slider(0.1, 1.0, value=0.8, step=0.1, label="میزان کاهش نویز", visible=False)
                    nr_enable.change(lambda x: gr.update(visible=x), inputs=nr_enable, outputs=nr_amount)
                
                with gr.Group():
                    ps_enable = gr.Checkbox(label="Pitch Shift", value=False)
                    ps_amount = gr.Slider(-12, 12, value=0, step=1, label="semitone", visible=False)
                    ps_enable.change(lambda x: gr.update(visible=x), inputs=ps_enable, outputs=ps_amount)
                
                with gr.Group():
                    eq_enable = gr.Checkbox(label="EQ Filter", value=False)
                    with gr.Column(visible=False) as eq_row:
                        eq_band_pl = gr.Dropdown(["bass", "mid", "treble"], value="bass", label="باند")
                        eq_gain_pl = gr.Slider(-15, 15, value=0, step=1, label="dB")
                    eq_enable.change(lambda x: gr.update(visible=x), inputs=eq_enable, outputs=eq_row)
                
                with gr.Group():
                    ts_enable = gr.Checkbox(label="Time Stretch", value=False)
                    ts_amount = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="ضریب سرعت", visible=False)
                    ts_enable.change(lambda x: gr.update(visible=x), inputs=ts_enable, outputs=ts_amount)
                
                pl_btn = gr.Button("▶ اجرای Pipeline", variant="primary")
            
            with gr.Column():
                pl_output      = gr.Audio(label="خروجی")
                pl_download_wav = gr.File(label="دانلود WAV")
                pl_download_mp3 = gr.File(label="دانلود MP3")
                pl_wave_before   = gr.Plot(label="نمودار قبل از پردازش")
                pl_wave_after    = gr.Plot(label="نمودار بعد از پردازش")               
                pl_status      = gr.Textbox(label="وضعیت", interactive=False)
        
        pl_btn.click(
            process_pipeline,
            inputs=[pl_mic, pl_upload, nr_enable, nr_amount, ps_enable, ps_amount, eq_enable, eq_band_pl, eq_gain_pl, ts_enable, ts_amount],
            outputs=[pl_output, pl_download_wav, pl_download_mp3, pl_wave_before,pl_wave_after , pl_status]
        )

    with gr.Tab("🎤 Vocal Isolation"):
        gr.Markdown("### جداسازی صدای خواننده از موزیک")
        with gr.Row():
            with gr.Column():
                vi_upload = gr.Audio(source="upload", type="filepath", label="آپلود فایل موزیک (WAV/MP3)")
                vi_btn    = gr.Button("▶ جداسازی", variant="primary")
            with gr.Column():
                vi_vocals      = gr.Audio(label="صدای خواننده (Vocals)")
                vi_instrumental = gr.Audio(label="موزیک بدون صدا (Instrumental)")
                vi_dl_vocals   = gr.File(label="دانلود Vocals")
                vi_dl_inst     = gr.File(label="دانلود Instrumental")
                vi_status      = gr.Textbox(label="وضعیت", interactive=False)
        vi_btn.click(
            process_vocal_isolation,
            inputs=[vi_upload],
            outputs=[vi_vocals, vi_instrumental, vi_dl_vocals, vi_dl_inst, vi_status]
        )
if __name__ == "__main__":
    app.launch()