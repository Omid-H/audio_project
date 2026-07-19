import sounddevice as sd
import numpy as np

def test_mic():
    print("Recording 3 seconds...")
    audio = sd.rec(int(3 * 44100), samplerate=44100, channels=1)
    sd.wait()
    print("Playing back...")
    sd.play(audio, 44100)
    sd.wait()
    print("Done!")

test_mic()