from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TourismSystem:
    """旅游系统状态"""

    visitors: float  # 游客数量
    environment: float  # 环境质量指数
    eco_investment: float  # 环境保护投资
    infra_cost: float  # 基础设施成本


class SustainableTourismModel:
    def __init__(self):
        # 模型参数
        self.k1 = 0.1  # 环保投资增益系数
        self.k2 = 0.05  # 游客环境损耗系数
        self.k3 = 0.02  # 自然恢复系数
        self.gamma = 0.3  # 满意度驱动的增长率
        self.delta = 0.1  # 环境恶化导致的衰减率
        self.T = 10  # 规划期限

        # 约束条件
        self.CO2_base = 1000  # 基准碳排放量
        self.waste_cap = 800  # 垃圾处理能力
        self.visitor_cap = 1.2e6  # 年游客上限

    def environment_dynamics(self, state: TourismSystem) -> float:
        """环境质量变化方程"""
        dE_dt = (
            self.k1 * state.eco_investment
            - self.k2 * state.visitors
            + self.k3 * self.get_natural_recovery_rate(state.eco_investment)
        )
        return dE_dt

    def visitor_dynamics(
        self, state: TourismSystem, satisfaction: float, E_min: float
    ) -> float:
        """游客数量变化方程"""
        V_next = state.visitors * (1 + self.gamma * satisfaction / 100) - self.delta * (
            E_min - state.environment
        )
        return max(0, min(V_next, self.visitor_cap))

    def get_natural_recovery_rate(self, eco_investment: float) -> float:
        """计算自然恢复速率"""
        base_rate = 5  # 基准恢复率
        return base_rate + 2 * (eco_investment / 1000)  # 每增加10%投资提升2%

    def calculate_emissions(
        self, visitors: float, per_capita_consumption: float
    ) -> float:
        """计算碳排放量"""
        emission_factor = 0.1  # 每单位消费的碳排放系数
        return visitors * per_capita_consumption * emission_factor

    def calculate_waste(self, visitors: float) -> float:
        """计算垃圾产生量"""
        waste_per_person = 0.5  # 人均垃圾产生量
        return visitors * waste_per_person

    def simulate_system(
        self,
        initial_state: TourismSystem,
        spending: float,  # 人均消费
        tax_rate: float,  # 人均税收
    ) -> List[TourismSystem]:
        """模拟系统演化"""
        states = [initial_state]

        for t in range(self.T):
            current = states[-1]

            # 计算环保投资和基础设施投资
            total_revenue = current.visitors * (spending + tax_rate)
            eco_investment = 0.4 * total_revenue
            infra_cost = 0.3 * total_revenue

            # 更新环境质量
            dE = self.environment_dynamics(current)
            new_E = max(0, min(100, current.environment + dE))

            # 更新游客数量
            tourist_satisfaction = 80  # 示例满意度值
            new_V = self.visitor_dynamics(current, tourist_satisfaction, 60)

            # 创建新状态
            new_state = TourismSystem(
                visitors=new_V,
                environment=new_E,
                eco_investment=eco_investment,
                infra_cost=infra_cost,
            )
            states.append(new_state)

        return states

    def evaluate_objectives(
        self, states: List[TourismSystem], spending: float, tax_rate: float
    ) -> Tuple[float, float, float]:
        """计算目标函数值"""
        # 经济收益
        total_revenue = sum(s.visitors * (spending + tax_rate) for s in states)
        total_cost = sum(s.infra_cost for s in states)
        economic_benefit = total_revenue - total_cost

        # 环境负担
        total_emissions = sum(
            self.calculate_emissions(s.visitors, spending) for s in states
        )
        total_waste = sum(self.calculate_waste(s.visitors) for s in states)
        environmental_burden = total_emissions + total_waste

        # 社会满意度 (简化计算)
        resident_satisfaction = 70  # 示例值
        tourist_satisfaction = 80  # 示例值
        social_satisfaction = sum(
            resident_satisfaction + tourist_satisfaction for _ in states
        )

        return economic_benefit, environmental_burden, social_satisfaction

    def check_constraints(
        self, states: List[TourismSystem], spending: float, tax_rate: float
    ) -> bool:
        """检查约束条件"""
        for state in states:
            # 环境承载力约束
            emissions = self.calculate_emissions(state.visitors, spending)
            waste = self.calculate_waste(state.visitors)
            if emissions > 0.7 * self.CO2_base or waste > 1.2 * self.waste_cap:
                return False

            # 社会容忍度约束
            if state.visitors > self.visitor_cap:
                return False

            # 经济可行性约束
            if spending < 200 or tax_rate > 50:
                return False

        return True


def main():
    # 创建模型实例
    model = SustainableTourismModel()

    # 初始状态
    initial_state = TourismSystem(
        visitors=500000,  # 初始游客数量
        environment=80,  # 初始环境质量
        eco_investment=1000000,  # 初始环保投资
        infra_cost=800000,  # 初始基础设施成本
    )

    # 模拟不同策略
    spending_options = [200, 300, 400]  # 人均消费方案
    tax_options = [20, 30, 40]  # 人均税收方案

    results = []
    for spending in spending_options:
        for tax in tax_options:
            states = model.simulate_system(
                initial_state=initial_state, spending=spending, tax_rate=tax
            )

            if model.check_constraints(states, spending, tax):
                objectives = model.evaluate_objectives(states, spending, tax)
                results.append((spending, tax, objectives))

    # 绘制结果
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for spending, tax, (f1, f2, f3) in results:
        ax.scatter(f1, f2, f3, marker="o")

    ax.set_xlabel("经济收益")
    ax.set_ylabel("环境负担")
    ax.set_zlabel("社会满意度")
    plt.title("可持续旅游发展多目标优化结果")
    plt.show()


if __name__ == "__main__":
    main()
