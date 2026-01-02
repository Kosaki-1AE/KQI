import numpy as np
import matplotlib.pyplot as plt

class FlowPointCycleModel:
    def __init__(self, flow_function, point_list, cycle_function):
        """
        flow_function: 時系列変化を表す関数（流れ）
        point_list: 各時点の観測点やイベント（点）
        cycle_function: 周期関数（周期）
        """
        self.flow = flow_function
        self.points = point_list
        self.cycle = cycle_function

    def simulate(self, t_range):
        """
        指定時間範囲でシミュレーションを行う
        """
        flow_values = [self.flow(t) for t in t_range]
        cycle_values = [self.cycle(t) for t in t_range]
        return flow_values, cycle_values

    def plot(self, t_range):
        flow_values, cycle_values = self.simulate(t_range)

        plt.figure(figsize=(10, 4))
        plt.plot(t_range, flow_values, label='Flow (流れ)', linestyle='-')
        plt.plot(t_range, cycle_values, label='Cycle (周期)', linestyle='--')
        for pt in self.points:
            plt.axvline(pt, color='r', linestyle=':', label='Point (点)' if pt == self.points[0] else "")
        plt.legend()
        plt.title("Flow × Point × Cycle")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()
