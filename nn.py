import os
import shutil
import subprocess
from transformers import pipeline
from pydub import AudioSegment
from pydub.silence import split_on_silence
import librosa

def convert_to_proper_wav(input_path, output_path):
    try:
        subprocess.run([
            'ffmpeg',
            '-i', input_path,
            '-c:a', 'pcm_s16le',
            output_path
        ], check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Ошибка конвертации: {e.stderr.decode()}")
        raise

def Transcribe_Audio(filename, task_id):
    convert_to_proper_wav(filename, f"fixed_{filename}")
    filename = f"fixed_{filename}"

    audio = AudioSegment.from_wav(filename)

    min_silence_len = 500
    max_chunk_len = 30 * 1000

    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=-40,
        keep_silence=200
    )

    if len(chunks) == 1 and len(chunks[0]) > max_chunk_len:
        chunks = [
            audio[i:i + max_chunk_len] 
            for i in range(0, len(audio), max_chunk_len)
        ]

    final_chunks = []
    current_chunk = AudioSegment.empty()

    for chunk in chunks:
        if len(current_chunk) + len(chunk) <= max_chunk_len:
            current_chunk += chunk
        else:
            if len(current_chunk) > 0:
                final_chunks.append(current_chunk)
            current_chunk = chunk

    if len(current_chunk) > 0:
        final_chunks.append(current_chunk)

    if os.path.exists(f"chunks{task_id}"):
        shutil.rmtree(f"chunks{task_id}")

    os.mkdir(f"chunks{task_id}")

    for i, chunk in enumerate(final_chunks):
        chunk.export(f"chunks{task_id}/{i}.wav", format="wav")
        #print(f"chunk_{i}.wav — {len(chunk)/1000:.1f} сек")


    pipe = pipeline("automatic-speech-recognition", model="Auttar/whisper-finetuned-shortened-medium", device=0)

    result = ""

    for i in range(len(final_chunks)):
        result += pipe(f"chunks{task_id}/{i}.wav")["text"]
        print(f"processed chunk {i} for task {task_id}")

    shutil.rmtree(f"chunks{task_id}")
    return result


result = Transcribe_Audio("audio1.wav", 1)
print(result)