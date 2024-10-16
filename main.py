import os
import wave
import re
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
from vosk import Model, KaldiRecognizer

model_path = "vosk-model"
output_folder = 'outputs'
output_wav_path = os.path.join(output_folder, 'output_audio.wav')
transcription_file = os.path.join(output_folder, 'transcription.txt')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

model = Model(model_path)
rec = KaldiRecognizer(model, 16000)

def select_microphone():
    audio = pyaudio.PyAudio()
    mic_count = audio.get_device_count()
    print("Доступные микрофоны:")
    for i in range(mic_count):
        info = audio.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"{i}: {info['name']}")
    mic_index = int(input("Выберите номер микрофона: "))
    return mic_index

mic_index = select_microphone()
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000, input_device_index=mic_index)
stream.start_stream()
wav_file = wave.open(output_wav_path, 'wb')
wav_file.setnchannels(1)
wav_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
wav_file.setframerate(16000)

def record_and_transcribe():
    print("Началась запись. Нажмите Ctrl+C для остановки...")
    transcription = []
    try:
        while True:
            data = stream.read(4000)
            wav_file.writeframes(data)
            if rec.AcceptWaveform(data):
                result = rec.Result()
                text = re.search(r'"text" : "(.*)"', result).group(1)
                transcription.append(text)
                print(f"Текст: {text}")
    except KeyboardInterrupt:
        print("Запись остановлена.")
        stream.stop_stream()
        stream.close()
        p.terminate()
        wav_file.close()

    return " ".join(transcription)

phonemes_dict = {
    'а': 'а', 'б': 'б', 'в': 'в', 'г': 'г', 'д': 'д', 'е': 'э', 'ё': 'о', 'ж': 'ж',
    'з': 'з', 'и': 'и', 'й': 'й', 'к': 'к', 'л': 'л', 'м': 'м', 'н': 'н', 'о': 'о',
    'п': 'п', 'р': 'р', 'с': 'с', 'т': 'т', 'у': 'у', 'ф': 'ф', 'х': 'х', 'ц': 'ц',
    'ч': 'ч', 'ш': 'ш', 'щ': 'щ', 'ы': 'ы', 'э': 'э', 'ю': 'у', 'я': 'а',
    'ь': '', 'ъ': ''
}

soft_consonants = {
    'б': 'б\'', 'в': 'в\'', 'г': 'г\'', 'д': 'д\'', 'ж': 'ж\'', 'з': 'з\'',
    'к': 'к\'', 'л': 'л\'', 'м': 'м\'', 'н': 'н\'', 'п': 'п\'', 'р': 'р\'',
    'с': 'с\'', 'т': 'т\'', 'ф': 'ф\'', 'х': 'х\'', 'ц': 'ц\'', 'ч': 'ч\'',
    'ш': 'ш\'', 'щ': 'щ\''
}

def text_to_phonemes(text):
    words = text.split()
    phonetic_text = []

    for word in words:
        phonetic_word = []
        for i, char in enumerate(word):
            if char in phonemes_dict:
                if i < len(word) - 1 and word[i + 1] == 'ь':
                    if char in soft_consonants:
                        phonetic_word.append(soft_consonants[char])
                else:
                    phonetic_word.append(phonemes_dict[char])
            else:
                phonetic_word.append(char)
        phonetic_text.append("".join(phonetic_word))

    return " ".join(phonetic_text)

def generate_spectrogram(audio_path):
    with wave.open(audio_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()
        samples = wf.readframes(n_frames)

    samples = np.frombuffer(samples, dtype=np.int16)

    if n_channels > 1:
        samples = samples[::n_channels]

    plt.figure(figsize=(12, 8))
    plt.specgram(samples, NFFT=2048, Fs=frame_rate, Fc=0, noverlap=128, cmap='inferno', sides='default', mode='default')
    plt.title('Спектрограмма')
    plt.ylabel('Частота (Гц)')
    plt.xlabel('Время (с)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'spectrogram.png'))
    plt.close()

transcribed_text = record_and_transcribe()
phonetic_transcription = text_to_phonemes(transcribed_text)

with open(transcription_file, 'w', encoding='utf-8') as f:
    f.write(f"Текст: {transcribed_text}\n")
    f.write(f"Фонетическая транскрипция: {phonetic_transcription}\n")

generate_spectrogram(output_wav_path)

print("Фонетическая транскрипция сохранена в:", transcription_file)
print("Спектрограмма сохранена в:", os.path.join(output_folder, 'spectrogram.png'))
