import time
import webbrowser

# 動画URL（例としてYouTubeのURL）
video_url = "https://www.youtube.com/watch?v=LmoMg0zVr6w"

# 10秒待つ
print("10秒タイマー開始…")
time.sleep(10)

# 動画を開く
print("動画再生します！")
webbrowser.open(video_url)