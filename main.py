import json
import subprocess
from urllib.parse import unquote, urlparse
import pika
import grpc
from concurrent import futures
import requests
import msu_logging_pb2
import msu_logging_pb2_grpc
from transformers import pipeline
import os
import shutil
from transformers import pipeline
from pydub import AudioSegment
from pydub.silence import split_on_silence
from moviepy import *

def ConvertToWAV(input_path, output_path):
    filename, file_extension = os.path.splitext(input_path)

    if file_extension.lower() == 'mp4':
        clip = VideoFileClip(input_path)
        audio = clip.audio
        audio.write_audiofile(output_path)
    else:
        convert_to_proper_wav(input_path,output_path)

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

def Transcribe_Audio(fileLink, task_id):
    print(f"Downloading audio {fileLink} for {task_id}")
    downloadedFile = requests.get(fileLink)

    parsed = urlparse(fileLink)
    basefilename = unquote(parsed.path.split("/")[-1])

    with open(basefilename, "wb") as f:
        f.write(downloadedFile.content)

    print(f"Converting audio {basefilename} for {task_id}")

    ConvertToWAV(basefilename, f"fixed_{os.path.basename(basefilename).split('.')[0]}.wav")
    filename = f"fixed_{os.path.basename(basefilename).split('.')[0]}.wav"


    print(f"Separating audio {filename} for {task_id}")

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


    print(f"Loading NN model for {task_id}")

    pipe = pipeline("automatic-speech-recognition", model="Auttar/whisper-finetuned-shortened-medium", device=0)

    result = ""

    for i in range(len(final_chunks)):
        result += pipe(f"chunks{task_id}/{i}.wav")["text"]
        print(f"processed chunk {i} for task {task_id}")

    shutil.rmtree(f"chunks{task_id}")
    os.remove(basefilename)
    os.remove(filename)

    print(f"Successfully transcribed audio for {task_id}")
    return result


backend_service_url = 'backend-service'
rabbit_url = 'rabbitmq'

class RabbitMQConsumer:
    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.URLParameters(f'amqp://admin:admin@{rabbit_url}:5672/')
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='transcribe_queue', durable=True)
        
        self.grpc_channel = grpc.insecure_channel(
        f'{backend_service_url}:50051',
        options=[
            ('grpc.connect_timeout_ms', 5000),  # 5 секунд на подключение
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
        ]
        )
        self.grpc_stub = msu_logging_pb2_grpc.TranscribeStub(self.grpc_channel)

    def callback(self, ch, method, properties, body):
        try:
            message = json.loads(body)
            task_id = message['TaskId']
            audio_file_link = message['AudioFileLink']
            
            print(f"Received task: {task_id}, audio file: {audio_file_link}")
            
            transcription_result = Transcribe_Audio(audio_file_link, task_id)
            
            grpc_response = self.grpc_stub.SendTranscribeResult(
                msu_logging_pb2.TranscribeResult(
                    success=True,
                    errorMessage="",
                    result=transcription_result,
                    taskId=task_id
                )
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
            print(f"gRPC response: {grpc_response}")
            
        except Exception as e:
            print(f"Error processing message: {e}")
            self.grpc_stub.SendTranscribeResult(
                msu_logging_pb2.TranscribeResult(
                    success=False,
                    errorMessage=f"Error processing message: {e}",
                    result="",
                    taskId=task_id
                )
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def start_consuming(self):
        self.channel.basic_consume(
            queue='transcribe_queue',
            on_message_callback=self.callback,
            auto_ack=False
        )
        print('Waiting for messages. To exit press CTRL+C')
        self.channel.start_consuming()

if __name__ == '__main__':
    try:
        consumer = RabbitMQConsumer()
        consumer.start_consuming()
    except KeyboardInterrupt:
        print("Consumer stopped by user")
    except Exception as e:
        print(f"Error: {e}")