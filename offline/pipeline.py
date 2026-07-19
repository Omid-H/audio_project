import numpy as np
from offline.pitch_shift import pitch_shift
from offline.eq_filter import apply_eq
from offline.noise_reduction import reduce_noise
from offline.time_stretch import time_stretch


def run_pipeline(audio: np.ndarray, sr: int, steps: list) -> np.ndarray:
    """
    پردازش زنجیره‌ای صدا
    
    Parameters:
    -----------
    audio : np.ndarray
        سیگنال صوتی نرمال‌شده (float32, -1 تا 1)
    sr : int
        Sample rate
    steps : list of dict
        هر dict یک مرحله پردازشه:
        {"type": "noise_reduction", "prop_decrease": 0.8}
        {"type": "pitch_shift", "n_steps": 3}
        {"type": "eq", "band": "bass", "gain_db": 5}
        {"type": "time_stretch", "rate": 1.2}
    
    Returns:
    --------
    np.ndarray : سیگنال پردازش‌شده
    """
    result = audio.copy()
    
    for step in steps:
        t = step.get("type")
        
        if t == "noise_reduction":
            prop = step.get("prop_decrease", 0.8)
            result = reduce_noise(result, sr, stationary=True, prop_decrease=prop)
        
        elif t == "pitch_shift":
            n_steps = step.get("n_steps", 0)
            if n_steps != 0:
                result = pitch_shift(result, sr, n_steps)
        
        elif t == "eq":
            band    = step.get("band", "bass")
            gain_db = step.get("gain_db", 0)
            if gain_db != 0:
                result = apply_eq(result, sr, band, gain_db)
        
        elif t == "time_stretch":
            rate = step.get("rate", 1.0)
            if rate != 1.0:
                result = time_stretch(result, rate)
        
        else:
            raise ValueError(f"Unknown step type: {t}")
    
    return result