# model_test.py の冒頭に以下を追加！
import numpy as np

from concept_base import FlowPointCycleModel

# 流れ：感情のなだらかな上昇と下降（tanh）
flow_func = lambda t: np.tanh((t - 5) / 2)

# 点：イベントが発生した時間（例：8時、12時、17時）
points = [2, 5, 8]

# 周期：一日の感情波（sin）
cycle_func = lambda t: np.sin(2 * np.pi * t / 10)

# モデル化
model = FlowPointCycleModel(flow_func, points, cycle_func)

# 可視化
t_range = np.linspace(0, 10, 200)
model.plot(t_range)
