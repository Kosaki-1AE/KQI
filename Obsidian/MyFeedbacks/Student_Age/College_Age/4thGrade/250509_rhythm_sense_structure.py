# 構造知モデルのコード化（強化版）：感覚の積分と微分をもとに次の行動を算出（3軸＋感情強度＋MediaPipe Pose＋音＆表情認識＋ビート同期＋精度強化＋推論モード分離）

import csv
import math
import time

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import sounddevice as sd
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression


class StructureKnowledgeModel:
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, epsilon=0.01):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

        self.history = []
        self.airflow_log = []
        self.prediction_log = []
        self.motion_vector_log = []
        self.emotion_log = []
        self.intensity_log = []

        self.regressor = LinearRegression()
        self.training_X = []  # 入力特徴量： [dF/dt, airflow, intensity]
        self.training_Y = []  # 出力： motion_vector（3軸）
        self.use_model_only = False  # モード切替フラグ：Trueなら学習済モデルのみで推論

    def add_action(self, action_value):
        self.history.append(action_value)

    def add_airflow(self, airflow_value):
        self.airflow_log.append(airflow_value)

    def add_emotion(self, emotion_name, intensity=1.0):
        self.emotion_log.append(emotion_name)
        self.intensity_log.append(intensity)

    def compute_feedback_derivative(self):
        if len(self.history) < 2:
            return 0.0
        return self.history[-1] - self.history[-2]

    def compute_equilibrium_error(self, n=5):
        recent = self.history[-n:] if len(self.history) >= n else self.history
        return abs(sum(recent))

    def get_emotion_vector(self, emotion_name):
        emotion_map = {
            'joy':      [0.1, 0.5, 0.8],
            'trust':    [0.2, 0.6, 0.7],
            'fear':     [0.3, 0.3, 0.1],
            'surprise': [0.7, 0.2, 0.2],
            'sadness':  [0.0, 0.2, 0.3],
            'disgust':  [0.1, 0.1, 0.4],
            'anger':    [0.9, 0.1, 0.1],
            'anticipation': [0.4, 0.3, 0.6],
        }
        return emotion_map.get(emotion_name, [0.3, 0.3, 0.3])

    def predict_next_action(self, current_action, current_airflow, emotion_name="neutral", intensity=1.0):
        dF_dt = self.compute_feedback_derivative()
        P = self.compute_equilibrium_error()

        if self.use_model_only and len(self.training_X) >= 5:
            motion_vector = self.regressor.predict([[dF_dt, current_airflow, intensity]])[0]
        else:
            next_action = (
                self.alpha * dF_dt +
                self.beta * current_airflow -
                self.gamma * P +
                self.delta * current_action
            )
            base_vector = self.get_emotion_vector(emotion_name)
            motion_vector = [next_action * w * intensity for w in base_vector]

            self.training_X.append([dF_dt, current_airflow, intensity])
            self.training_Y.append(motion_vector)

        self.prediction_log.append(current_action)
        self.motion_vector_log.append(motion_vector)
        return motion_vector

    def train_model(self):
        if len(self.training_X) >= 5:
            self.regressor.fit(self.training_X, self.training_Y)

    def predict_from_model(self, dF_dt, airflow, intensity):
        if len(self.training_X) >= 5:
            return self.regressor.predict([[dF_dt, airflow, intensity]])[0]
        else:
            return [0.0, 0.0, 0.0]

    def plot_logs(self):
        plt.figure(figsize=(12, 6))
        t = range(len(self.motion_vector_log))
        robot = [v[0] for v in self.motion_vector_log]
        wave = [v[1] for v in self.motion_vector_log]
        rhythm = [v[2] for v in self.motion_vector_log]
        plt.plot(t, robot, label="Robot")
        plt.plot(t, wave, label="Wave")
        plt.plot(t, rhythm, label="Rhythm")
        plt.title("3軸ベクトルログ（ロボット／ウェーブ／リズム）")
        plt.xlabel("時刻 t")
        plt.ylabel("出力")
        plt.legend()
        plt.grid(True)
        plt.show()

    def export_training_data(self, filename="training_data.csv"):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["action", "airflow", "emotion", "intensity", "robot", "wave", "rhythm"])
            for a, c, e, i, v in zip(self.history, self.airflow_log, self.emotion_log, self.intensity_log, self.motion_vector_log):
                writer.writerow([a, c, e, i] + v)

    def set_model_mode(self, use_model):
        self.use_model_only = use_model
