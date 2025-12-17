# EmotionMotion: 恋愛空気感支援モード（感情 × 空気感 × 共感動作 推定＋学習機能）

import numpy as np
from sklearn.linear_model import LogisticRegression


class EmotionRelationshipAdvisor:
    def __init__(self):
        self.emotion_log = []  # 感情カテゴリ
        self.intensity_log = []  # 感情強度（0〜1）
        self.rhythm_log = []  # リズム感（0〜1）
        self.label_log = []    # 実際の相手の反応ラベル（0:悪い, 1:良い）
        self.model = LogisticRegression()
        self.trained = False

    def analyze_state(self, emotion_name, intensity, rhythm):
        emotion_vector = self.encode_emotion(emotion_name)
        features = emotion_vector + [intensity, rhythm]

        # 推論（学習済みなら）
        if self.trained:
            prob = self.model.predict_proba([features])[0][1]
            prediction_feedback = f"🔮 学習による予測: 好感度 {prob*100:.1f}%\n"
        else:
            prediction_feedback = "📚 モデルはまだ十分に学習されていません。\n"

        # 人的アドバイス
        advice = ""
        if emotion_name in ["joy", "trust"]:
            if intensity > 0.6:
                advice += "✨ 今は安心感が強く、好意的な空気感があります。自分らしく話しかけてOK。\n"
            else:
                advice += "😊 穏やかな雰囲気です。焦らず相手のペースに合わせよう。\n"
        elif emotion_name in ["sadness", "disgust"]:
            advice += "💤 今はややネガティブな感情がありそう。声のトーンや間合いに注意。\n"
        elif emotion_name == "anger":
            advice += "⚠️ 相手がイライラしているかも。静かな共感とリアクションを意識しよう。\n"
        elif emotion_name == "surprise":
            advice += "😮 何かに驚いたような気配。相手の反応を優しく受け止めてあげよう。\n"
        elif emotion_name == "fear":
            advice += "😟 不安感が見えるかも。言葉選びや表情で安心感を出してみて。\n"

        if rhythm > 0.7:
            advice += "🎵 会話のテンポはやや速め。軽やかにリアクションするのが吉。"
        elif rhythm < 0.3:
            advice += "🕊 ゆっくりしたリズム。落ち着いた声・目線を意識しよう。"
        else:
            advice += "🎶 自然なテンポ感。無理に合わせず心地よく会話できそう。"

        return prediction_feedback + advice

    def encode_emotion(self, emotion_name):
        emotions = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
        return [1 if emotion_name == emo else 0 for emo in emotions]

    def record_feedback(self, emotion_name, intensity, rhythm, label):
        # label: 1（好感） or 0（イマイチ）など
        emotion_vector = self.encode_emotion(emotion_name)
        features = emotion_vector + [intensity, rhythm]
        self.emotion_log.append(emotion_name)
        self.intensity_log.append(intensity)
        self.rhythm_log.append(rhythm)
        self.label_log.append(label)

        if len(self.label_log) >= 5:
            X = [self.encode_emotion(e) + [i, r] for e, i, r in zip(self.emotion_log, self.intensity_log, self.rhythm_log)]
            y = self.label_log
            self.model.fit(X, y)
            self.trained = True

# 使用例
advisor = EmotionRelationshipAdvisor()
print(advisor.analyze_state("trust", 0.8, 0.5))
advisor.record_feedback("trust", 0.8, 0.5, 1)
advisor.record_feedback("sadness", 0.4, 0.2, 0)
advisor.record_feedback("joy", 0.9, 0.9, 1)
advisor.record_feedback("anger", 0.7, 0.6, 0)
advisor.record_feedback("fear", 0.3, 0.4, 1)
print("\n" + advisor.analyze_state("joy", 0.9, 0.6))
