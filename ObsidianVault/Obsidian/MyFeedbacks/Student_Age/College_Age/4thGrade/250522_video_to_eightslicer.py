import os
import subprocess
import librosa
from pydub import AudioSegment
import math

# === ユーザーが設定すべきパス ===
input_video = "recording.mp4"  # 処理したい画録動画ファイル
ffmpeg_path = "C:\Users\society5\Downloads\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"  # ←ここを自分のffmpeg.exeの実パスに！
output_dir = "split_chunks"
temp_audio = "temp_audio.wav"

# === ステップ1: ffmpegで音声抽出 ===
if not os.path.exists(input_video):
    raise FileNotFoundError(f"入力動画が見つからんよ: {input_video}")

subprocess.run([
    ffmpeg_path, "-i", input_video,
    "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
    temp_audio
], check=True)

# === ステップ2: BPM検出 ===
y, sr = librosa.load(temp_audio)
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
print(f"テンポ検出完了: {tempo:.2f} BPM")

# === ステップ3: 4エイト（32拍）ごとに分割 ===
beat_duration = 60 / tempo
chunk_duration = beat_duration * 32  # 秒

audio = AudioSegment.from_wav(temp_audio)
num_chunks = math.floor(len(audio) / (chunk_duration * 1000))

os.makedirs(output_dir, exist_ok=True)

for i in range(num_chunks):
    start_ms = i * chunk_duration * 1000
    end_ms = (i + 1) * chunk_duration * 1000
    chunk = audio[start_ms:end_ms]
    chunk.export(f"{output_dir}/chunk_{i+1:02d}.mp3", format="mp3")

print(f"{num_chunks} 個の4エイト分割ファイルを {output_dir} に保存したで！")

# 一時ファイルを削除（必要なら）
os.remove(temp_audio)
