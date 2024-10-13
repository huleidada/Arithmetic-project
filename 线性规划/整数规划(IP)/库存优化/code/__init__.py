import pulp

# 数据
warehouses = [0, 1, 2]  # 仓库编号
customers = [0, 1, 2, 3, 4]  # 客户编号

# 客户需求量 (d_j) : 每个客户的需求量
demand = [100, 200, 150, 175, 120]

# 各仓库建设成本 (f_i) : 每个仓库的建设固定成本
construction_cost = [5000, 6000, 4000]

# 各仓库的运输成本 (c_ij) : 仓库到客户的单位运输成本
transport_cost = [
    [2, 4, 5, 3, 6],  # 仓库0到客户的运输成本
    [3, 2, 4, 5, 3],  # 仓库1到客户的运输成本
    [4, 3, 2, 6, 5],  # 仓库2到客户的运输成本
]

# 仓库容量 (C_i) : 每个仓库的库存容量
warehouse_capacity = [300, 250, 200]

# 定义问题：目标是最小化运输成本与建设成本的和
problem = pulp.LpProblem("Inventory_Distribution_Optimization", pulp.LpMinimize)

# 决策变量：运输量 x_ij (从仓库i到客户j的运输量)
# x_ij 必须是整数，代表仓库i向客户j运输的数量
x = pulp.LpVariable.dicts("x", (warehouses, customers), lowBound=0, cat='Integer')

# 决策变量：仓库是否建设 y_i (仓库i是否建设)
# y_i 是二进制变量，表示仓库i是否建设，1表示建设，0表示不建设
y = pulp.LpVariable.dicts("y", warehouses, cat='Binary')

# 目标函数：最小化总成本
# 目标函数包括两部分：运输成本和建设成本
# 总成本 Z = sum(c_ij * x_ij) + sum(f_i * y_i)
problem += pulp.lpSum(transport_cost[i][j] * x[i][j] for i in warehouses for j in customers) + pulp.lpSum(
    construction_cost[i] * y[i] for i in warehouses), "Total Cost"

# 约束条件：

# 1. 每个客户的需求必须满足
# 需求约束：sum(x_ij) = d_j (每个客户j的需求d_j必须被完全满足)
for j in customers:
    problem += pulp.lpSum(x[i][j] for i in warehouses) == demand[j], f"Demand_{j}"

# 2. 仓库库存量不能超过其容量
# 仓库容量约束：sum(x_ij) <= C_i * y_i (仓库i的库存不能超过其最大容量C_i，只有在y_i为1时才考虑)
for i in warehouses:
    problem += pulp.lpSum(x[i][j] for j in customers) <= warehouse_capacity[i] * y[i], f"Capacity_{i}"

# 3. 只有在仓库建设的情况下，才会有运输量
# 运输约束：x_ij <= M * y_i (只有在仓库i被建设（y_i = 1）时，x_ij才大于0；M为一个大的常数，确保y_i为0时x_ij为0)
for i in warehouses:
    for j in customers:
        problem += x[i][j] <= 1000 * y[i], f"WarehouseOpen_{i}_Customer_{j}"

# 求解问题
problem.solve()

# 打印结果
if pulp.LpStatus[problem.status] == 'Optimal':
    print("找到最优解:")

    # 打印每个仓库是否建设
    for i in warehouses:
        print(f"仓库 {i} 建设: {pulp.value(y[i])}")

    # 打印每个仓库到每个客户的运输量
    for i in warehouses:
        for j in customers:
            if pulp.value(x[i][j]) > 0:
                print(f"从仓库 {i} 到客户 {j} 运输 {pulp.value(x[i][j])} 单位")

    print("总成本:", pulp.value(problem.objective))
else:
    print("未找到最优解。")
