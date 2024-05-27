from IPython.display import Audio, display, Javascript
import numpy as np
import simpleaudio as sa

def playSound():
    # Generate a 1-second sine wave at 440 Hz
    sample_rate = 44100  # samples per second
    duration = 1  # seconds
    frequency = 440  # Hz (A4 note)

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * frequency * t)
    waveform_integers = np.int16(waveform * 32767)
    
    # Play the sound
    play_obj = sa.play_buffer(waveform_integers, 1, 2, sample_rate)
    play_obj.wait_done()

def showPopup():
    display(Javascript('alert("Task Completed Successfully!")'))

def onCompletion():
    playSound()
    showPopup()
