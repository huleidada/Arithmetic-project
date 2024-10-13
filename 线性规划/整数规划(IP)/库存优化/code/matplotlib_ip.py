import pulp
import matplotlib.pyplot as plt
import numpy as np

# 数据：仓库、客户、需求、成本等
warehouses = [0, 1, 2]  # 仓库编号
customers = [0, 1, 2, 3, 4]  # 客户编号
demand = [100, 200, 150, 175, 120]  # 客户需求量 (d_j)
construction_cost = [5000, 6000, 4000]  # 仓库建设固定成本 (f_i)
transport_cost = [
    [2, 4, 5, 3, 6],  # 仓库0到客户的单位运输成本 (c_ij)
    [3, 2, 4, 5, 3],  # 仓库1到客户的单位运输成本 (c_ij)
    [4, 3, 2, 6, 5],  # 仓库2到客户的单位运输成本 (c_ij)
]
warehouse_capacity = [300, 250, 200]  # 仓库容量 (C_i)

# 定义问题：目标是最小化运输成本与建设成本的和
problem = pulp.LpProblem("Inventory_Distribution_Optimization", pulp.LpMinimize)

# 决策变量：运输量 x_ij (从仓库i到客户j的运输量)，必须为整数
# x_ij ≥ 0，表示仓库i向客户j运输的数量
x = pulp.LpVariable.dicts("x", (warehouses, customers), lowBound=0, cat='Integer')

# 决策变量：仓库是否建设 y_i (仓库i是否建设)，必须为0或1（二进制变量）
# y_i ∈ {0, 1}, 1表示建设，0表示不建设
y = pulp.LpVariable.dicts("y", warehouses, cat='Binary')

# 目标函数：最小化总成本（运输成本 + 仓库建设成本）
# 目标函数公式：
# Z = sum(c_ij * x_ij) + sum(f_i * y_i)
# Z = ∑ ∑ c_ij * x_ij + ∑ f_i * y_i
# 第一部分是运输成本：∑ ∑ c_ij * x_ij，表示从仓库i到客户j的运输成本总和。
# 第二部分是建设成本：∑ f_i * y_i，表示建设仓库的固定成本总和。
problem += pulp.lpSum(transport_cost[i][j] * x[i][j] for i in warehouses for j in customers) + pulp.lpSum(
    construction_cost[i] * y[i] for i in warehouses), "Total Cost"

# 约束条件：

# 1. 满足每个客户的需求
# 需求约束公式：
# sum(x_ij) = d_j (每个客户j的需求d_j必须被完全满足)
# 即：对于每个客户j，仓库i向客户j运输的总量必须等于客户的需求量d_j。
for j in customers:
    problem += pulp.lpSum(x[i][j] for i in warehouses) == demand[j], f"Demand_{j}"

# 2. 仓库容量约束
# 仓库容量约束公式：
# sum(x_ij) <= C_i * y_i (仓库i的库存不能超过其最大容量C_i，只有在y_i为1时才考虑)
# 只有当仓库i被建设（y_i=1）时，才能运输x_ij个单位的货物。如果仓库i未建设（y_i=0），则没有运输量。
for i in warehouses:
    problem += pulp.lpSum(x[i][j] for j in customers) <= warehouse_capacity[i] * y[i], f"Capacity_{i}"

# 3. 只有在仓库建设的情况下，才会有运输量
# 运输约束公式：
# x_ij <= M * y_i (只有在仓库i被建设（y_i = 1）时，x_ij才大于0；M为一个大的常数，确保y_i为0时x_ij为0)
# 该约束确保当仓库未建设时，不会有运输量。
for i in warehouses:
    for j in customers:
        problem += x[i][j] <= 1000 * y[i], f"WarehouseOpen_{i}_Customer_{j}"

# 求解问题
problem.solve()

# 打印结果
if pulp.LpStatus[problem.status] == 'Optimal':
    print("找到最优解：")

    # 打印每个仓库是否建设
    for i in warehouses:
        print(f"仓库 {i} 建设状态: {pulp.value(y[i])}")

    # 打印每个仓库到每个客户的运输量
    for i in warehouses:
        for j in customers:
            if pulp.value(x[i][j]) > 0:
                print(f"从仓库 {i} 向客户 {j} 运输 {pulp.value(x[i][j])} 单位货物")

    # 打印总成本
    print("总成本:", pulp.value(problem.objective))

    # 可视化部分：

    # 1. 绘制仓库建设情况
    warehouse_construction = [pulp.value(y[i]) for i in warehouses]
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.bar(warehouses, warehouse_construction, color=['green' if v == 1 else 'red' for v in warehouse_construction])
    plt.xlabel('仓库编号')
    plt.ylabel('建设状态 (1: 已建设, 0: 未建设)')
    plt.title('仓库建设状态')

    # 2. 绘制每个仓库到客户的运输量
    for i in warehouses:
        transport_amount = [pulp.value(x[i][j]) for j in customers]
        plt.subplot(2, 1, 2)
        plt.bar(customers, transport_amount, label=f'仓库 {i}')

    plt.xlabel('客户编号')
    plt.ylabel('运输量')
    plt.title('仓库到客户的运输量')
    plt.legend()

    plt.tight_layout()
    plt.show()

else:
    print("未找到最优解。")
