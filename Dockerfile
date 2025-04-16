# Используем официальный образ Python
FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    && rm -rf /var/lib/apt/lists/*
# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .

RUN pip install --upgrade pip
# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install google
RUN pip install protobuf
RUN pip install torch==2.6.0
RUN pip install audioop

# Копируем всё остальное
COPY . .

# Настраиваем запуск нашего приложения
CMD ["python", "main.py"]