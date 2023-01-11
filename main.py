import os
from typing import NewType, TypeVar
from logging import basicConfig, getLogger, DEBUG, ERROR
from scipy.io import wavfile
from os import getenv
from dotenv import load_dotenv
import librosa
from librosa.display import specshow
import numpy as np
from matplotlib import pyplot
from IPython.display import Audio


load_dotenv('.env')
basicConfig(level='INFO', filename=None)
_logger = getLogger('main.py -->> ')

_FILE_NAME: str = getenv('TEST_FILE_NAME', 'test.mp3')
_TEST_FILE_PATH: str = os.path.abspath(_FILE_NAME)


def _percussive(data, rt):
    _, percussive = librosa.decompose.hpss(librosa.stft(data))
    rp = np.max(np.abs(rt))
    y_percusive = librosa.istft(percussive)
    return Audio(data=y_percusive, rate=rp)
    
 
def _harmonic(data, rt):
    harmonic, _ = librosa.decompose.hpss(librosa.stft(data))
    rp = np.max(np.abs(rt))
    y_harmonic = librosa.istft(harmonic)
    return Audio(data=y_harmonic, rate=rp)
    

def load_audio_file(file_path: str = _TEST_FILE_PATH):
    return librosa.load(_TEST_FILE_PATH, duration=13, offset=15)


def main(perc=False) -> None:
    data, rate = load_audio_file()
    if perc: 
        _logger.info("Percussive sample creation")
        return _percussive(data=data, rt=rate)
    _logger.info("Harmonic sample creation")
    return _harmonic(data=data, rt=rate)


if __name__ == '__main__':
    print(f"  {main(perc=True)}")
    print(f"  {main()}")