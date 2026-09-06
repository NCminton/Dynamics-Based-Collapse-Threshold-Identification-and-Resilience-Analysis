# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 15:52:51 2026

@author: mengxueyu
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from tqdm import tqdm

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# 设置mathtext字体为stix（类似于Times New Roman）
plt.rcParams['mathtext.fontset'] = 'stix'
font_size = 36

# 设置路径（已修改为去边的路径）
data_dir = r"E:\论文写作\23 供应链 系统动力学\2 数据\加权矩阵"
output_dir = r"E:\论文写作\35 TRE\代码\2 去边\22 随机方式\数据"
figure_dir = r"E:\论文写作\35 TRE\代码\2 去边\22 随机方式\作图"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

# 1. 读取2023年网络数据并立即对权重进行对数变换
def load_2023_network():
    """加载2023年网络数据，构建有向加权图，并对权重进行对数变换"""
    file_2023 = os.path.join(data_dir, "edgelist_2023.csv")
    df_2023 = pd.read_csv(file_2023)
    
    G = nx.DiGraph()
    for _, row in df_2023.iterrows():
        # 对权重进行对数变换 log10(x+1)
        log_weight = np.log(row['weight'] + 1)
        G.add_edge(row['source'], row['target'], weight=log_weight)
    
    return G

# 2. 构建网络并提取最大强连通分量
G = load_2023_network()
print(f"原始网络节点数: {G.number_of_nodes()}")
print(f"原始网络边数: {G.number_of_edges()}")

# 提取最大强连通分量
strongly_connected_components = list(nx.strongly_connected_components(G))
largest_scc = max(strongly_connected_components, key=len)
G_lcc = G.subgraph(largest_scc).copy()
G_lcc.remove_edges_from(nx.selfloop_edges(G_lcc))

print(f"最大强连通分量节点数: {G_lcc.number_of_nodes()}")
print(f"最大强连通分量边数: {G_lcc.number_of_edges()}")

# 计算每个国家的进出口量 - 使用对数变换后的权重
node_in_volumes = {}
node_out_volumes = {}
node_trade_volumes = {}

for node in G_lcc.nodes():
    in_volume = sum([G_lcc[u][node]['weight'] for u in G_lcc.predecessors(node)])
    out_volume = sum([G_lcc[node][v]['weight'] for v in G_lcc.successors(node)])
    node_in_volumes[node] = in_volume
    node_out_volumes[node] = out_volume
    node_trade_volumes[node] = in_volume + out_volume

# 输出对数变换后的贸易量范围
in_volumes = list(node_in_volumes.values())
out_volumes = list(node_out_volumes.values())
volumes = list(node_trade_volumes.values())

print(f"进口量范围: [{min(in_volumes):.2f}, {max(in_volumes):.2f}]")
print(f"出口量范围: [{min(out_volumes):.2f}, {max(out_volumes):.2f}]")
print(f"总贸易量范围: [{min(volumes):.2f}, {max(volumes):.2f}]")

# 构建进口矩阵E和出口矩阵G
nodes = list(G_lcc.nodes())
n = len(nodes)

# 创建节点到索引的映射
node_to_index = {node: i for i, node in enumerate(nodes)}

# 进口矩阵E: E[i,j]表示从j到i的进口权重
# 出口矩阵G: G[i,j]表示从i到j的出口权重
E = np.zeros((n, n))  # 进口矩阵
G_matrix = np.zeros((n, n))  # 出口矩阵（重命名为G_matrix避免与网络G冲突）

for i, node_i in enumerate(nodes):
    for j, node_j in enumerate(nodes):
        if i != j and G_lcc.has_edge(node_j, node_i):  # 注意方向：从j到i是进口
            E[i, j] = G_lcc[node_j][node_i]['weight']
        if i != j and G_lcc.has_edge(node_i, node_j):  # 从i到j是出口
            G_matrix[i, j] = G_lcc[node_i][node_j]['weight']

print(f"进口矩阵E非零元素数量: {np.count_nonzero(E)}")
print(f"出口矩阵G非零元素数量: {np.count_nonzero(G_matrix)}")

# 计算平均参数
x_in_values = np.array([node_in_volumes[node] for node in nodes])
x_out_values = np.array([node_out_volumes[node] for node in nodes])
x_values = np.array([node_trade_volumes[node] for node in nodes])

x_in_mean = np.mean(x_in_values)  # 平均进口量
x_out_mean = np.mean(x_out_values)  # 平均出口量
x_mean = np.mean(x_values)   # 平均贸易量
C1 = x_in_mean
C2 = x_out_mean

print(f"平均进口量 <x_in>: {x_in_mean}")
print(f"平均出口量 <x_out>: {x_out_mean}")
print(f"平均贸易量 <x>: {x_mean}")

# 3. 求解参数B - 使用改进的降维方程
def equation_for_B(B):
    """改进的降维韧性动力学方程 - 分母改为x_mean"""
    alpha = x_in_mean
    beta = x_out_mean
    return B * x_mean + alpha * (x_in_mean**2 / (x_in_mean**2 + C1)) + beta * (x_out_mean**2 / (x_out_mean**2 + C2))

# 使用数值方法求解B
B_solution = fsolve(equation_for_B, -0.1)[0]
print(f"求解得到的参数 B (降维方程): {B_solution}")

# 4. 计算每个节点的B_i
def compute_B_i_for_each_node(nodes, node_in_volumes, node_out_volumes, E, G_matrix, x_mean):
    """计算每个节点的B_i值 - 分母改为x_mean"""
    n = len(nodes)
    B_i_values = np.zeros(n)
    
    for i, node in enumerate(nodes):  
        # 获取节点i的进口量和出口量
        x_in_i = node_in_volumes[node]
        x_out_i = node_out_volumes[node]
        x_i = x_in_i + x_out_i
        
        if x_i == 0:
            B_i_values[i] = 0
            continue
            
        # 计算节点i的进口交互项和出口交互项 - 分母改为x_mean
        import_interaction = 0
        export_interaction = 0
        
        for j in range(n):
            if i != j:
                # 进口交互：从j到i的进口
                if E[i, j] > 0:
                    x_in_j = node_in_volumes[nodes[j]]
                    import_interaction += E[i, j] * (x_in_j**2 / (x_in_j**2 + C1))
                
                # 出口交互：从i到j的出口
                if G_matrix[i, j] > 0:
                    x_out_j = node_out_volumes[nodes[j]]
                    export_interaction += G_matrix[i, j] * (x_out_j**2 / (x_out_j**2 + C2))
        
        # 求解B_i：B_i * x_i + import_interaction + export_interaction = 0
        if x_i != 0:
            B_i_values[i] = -(import_interaction + export_interaction) / x_i
        else:
            B_i_values[i] = 0
    
    return B_i_values

# 计算每个节点的B_i - 分母改为x_mean
B_i_values = compute_B_i_for_each_node(nodes, node_in_volumes, node_out_volumes, E, G_matrix, x_mean)
B_mean = np.mean(B_i_values)
print(f"B_i的平均值: {B_mean}")
print(f"B_i的范围: [{np.min(B_i_values):.6f}, {np.max(B_i_values):.6f}]")

# 保存B_i值
B_i_df = pd.DataFrame({
    'Node': nodes,
    'B_i': B_i_values,
    'In_Volume': [node_in_volumes[node] for node in nodes],
    'Out_Volume': [node_out_volumes[node] for node in nodes],
    'Total_Volume': [node_trade_volumes[node] for node in nodes]
})
B_i_file = os.path.join(output_dir, "B_i_values.csv")
B_i_df.to_csv(B_i_file, index=False)
print(f"B_i值已保存至: {B_i_file}")

# ================ 修改部分1：生成随机去边顺序 ================
# 获取所有边及其权重
edges = list(G_lcc.edges(data=True))
edge_weights = [edge[2]['weight'] for edge in edges]
total_edge_weight = sum(edge_weights)

print(f"总边权重: {total_edge_weight}")
print(f"边数: {len(edges)}")

# 生成随机去边顺序
random_removal_order_edges = edges.copy()
np.random.shuffle(random_removal_order_edges)

# 保存随机去边顺序
edge_removal_order_df = pd.DataFrame({
    'Removal_Order': range(1, len(random_removal_order_edges) + 1),
    'Source': [edge[0] for edge in random_removal_order_edges],
    'Target': [edge[1] for edge in random_removal_order_edges],
    'Weight': [edge[2]['weight'] for edge in random_removal_order_edges]
})
removal_order_file = os.path.join(output_dir, "random_edge_removal_order.csv")
edge_removal_order_df.to_csv(removal_order_file, index=False)
print(f"随机去边顺序已保存至: {removal_order_file}")

# 5. 改进的韧性分析 - 基于理论解析的方法（使用平均B）- 修改为去边
def theoretical_robustness_analysis_edge_removal(G_lcc, nodes, node_in_volumes, node_out_volumes, B, x_mean, removal_order_edges):
    """改进的理论解析韧性分析 - 使用进口和出口分离的动力学，分母改为x_mean，随机去边"""
    theoretical_results_in = []  # 进口部分的稳态解
    theoretical_results_out = []  # 出口部分的稳态解
    theoretical_results = []  # 总贸易量的稳态解
    steady_state_solutions_in = []  # 进口部分的稳态解
    steady_state_solutions_out = []  # 出口部分的稳态解
    steady_state_solutions = []  # 总贸易量的稳态解
    alpha_values = []  # 存储每个q值对应的alpha
    beta_values = []  # 存储每个q值对应的beta
    q_values = []  # 存储每个移除步骤的q值
    
    # 初始网络
    G_current = G_lcc.copy()
    
    # 计算初始权重比例
    removed_weight = 0
    
    # 初始状态：没有边被移除
    q = 0.0
    q_values.append(q)
    
    # 计算当前剩余网络的平均进口量和平均出口量
    if G_current.number_of_nodes() > 0:
        current_in_volumes = [sum([G_current[u][node]['weight'] for u in G_current.predecessors(node)]) 
                              for node in G_current.nodes()]
        current_out_volumes = [sum([G_current[node][v]['weight'] for v in G_current.successors(node)]) 
                               for node in G_current.nodes()]
        alpha_current = np.mean(current_in_volumes) if current_in_volumes else 0
        beta_current = np.mean(current_out_volumes) if current_out_volumes else 0
    else:
        alpha_current = 0
        beta_current = 0
    
    alpha_values.append(alpha_current)
    beta_values.append(beta_current)
    
    # 分别求解进口和出口的稳态方程 - 分母改为x_mean
    def steady_state_eq_in(x_in):
        return B * x_in + alpha_current * (x_in**2 / (x_in**2 + C1))
    
    def steady_state_eq_out(x_out):
        return B * x_out + beta_current * (x_out**2 / (x_out**2 + C2))
    
    # 寻找进口部分的非零解
    if alpha_current > 0:
        # 尝试多个初始值来找到稳定解
        solutions_in = []
        for initial_guess in [0.1, x_in_mean, 2*x_in_mean, 5*x_in_mean]:
            try:
                x_solution = fsolve(steady_state_eq_in, initial_guess)[0]
                if x_solution > 0:  # 只考虑正解
                    solutions_in.append(x_solution)
            except:
                continue
        
        if solutions_in:
            # 选择最大的正解（对应稳定状态）
            x_solution_in = max(solutions_in)
            steady_state_solutions_in.append(x_solution_in)
            theoretical_results_in.append(x_solution_in)
        else:
            steady_state_solutions_in.append(0)
            theoretical_results_in.append(0)
    else:
        steady_state_solutions_in.append(0)
        theoretical_results_in.append(0)
    
    # 寻找出口部分的非零解
    if beta_current > 0:
        # 尝试多个初始值来找到稳定解
        solutions_out = []
        for initial_guess in [0.1, x_out_mean, 2*x_out_mean, 5*x_out_mean]:
            try:
                x_solution = fsolve(steady_state_eq_out, initial_guess)[0]
                if x_solution > 0:  # 只考虑正解
                    solutions_out.append(x_solution)
            except:
                continue
        
        if solutions_out:
            # 选择最大的正解（对应稳定状态）
            x_solution_out = max(solutions_out)
            steady_state_solutions_out.append(x_solution_out)
            theoretical_results_out.append(x_solution_out)
        else:
            steady_state_solutions_out.append(0)
            theoretical_results_out.append(0)
    else:
        steady_state_solutions_out.append(0)
        theoretical_results_out.append(0)
    
    # 总贸易量是进口和出口之和
    total_solution = steady_state_solutions_in[-1] + steady_state_solutions_out[-1]
    steady_state_solutions.append(total_solution)
    theoretical_results.append(total_solution)
    
    print(f"初始状态 - alpha: {alpha_current:.4f}, beta: {beta_current:.4f}")
    print(f"进口解: {steady_state_solutions_in[-1]:.4f}, 出口解: {steady_state_solutions_out[-1]:.4f}, 总解: {total_solution:.4f}")
    
    # 按照随机顺序逐条边移除
    for step, (u, v, attr) in enumerate(removal_order_edges):
        if not G_current.has_edge(u, v):
            continue
            
        # 从当前网络中移除该边
        edge_weight = G_current[u][v]['weight']
        G_current.remove_edge(u, v)
        
        # 更新已移除的权重
        removed_weight += edge_weight
        
        # 计算当前移除率q（基于边权重）
        q = removed_weight / total_edge_weight
        q_values.append(q)
        
        # 计算当前剩余网络的平均进口量和平均出口量
        if G_current.number_of_nodes() > 0:
            current_in_volumes = [sum([G_current[u][node]['weight'] for u in G_current.predecessors(node)]) 
                                  for node in G_current.nodes()]
            current_out_volumes = [sum([G_current[node][v]['weight'] for v in G_current.successors(node)]) 
                                   for node in G_current.nodes()]
            alpha_current = np.mean(current_in_volumes) if current_in_volumes else 0
            beta_current = np.mean(current_out_volumes) if current_out_volumes else 0
        else:
            alpha_current = 0
            beta_current = 0
        
        alpha_values.append(alpha_current)
        beta_values.append(beta_current)
        
        # 分别求解进口和出口的稳态方程 - 分母改为x_mean
        def steady_state_eq_in(x_in):
            return B * x_in + alpha_current * (x_in**2 / (x_in**2 + C1))
        
        def steady_state_eq_out(x_out):
            return B * x_out + beta_current * (x_out**2 / (x_out**2 + C2))
        
        # 寻找进口部分的非零解
        if alpha_current > 0:
            # 尝试多个初始值来找到稳定解
            solutions_in = []
            for initial_guess in np.linspace(0, 1000, 1000):
                try:
                    x_solution = fsolve(steady_state_eq_in, initial_guess)[0]
                    if x_solution > 0:  # 只考虑正解
                        solutions_in.append(x_solution)
                except:
                    continue
            
            if solutions_in:
                # 选择最大的正解（对应稳定状态）
                x_solution_in = max(solutions_in)
                steady_state_solutions_in.append(x_solution_in)
                theoretical_results_in.append(x_solution_in) 
            else:
                steady_state_solutions_in.append(0)
                theoretical_results_in.append(0)
        else:
            steady_state_solutions_in.append(0)
            theoretical_results_in.append(0)
        
        # 寻找出口部分的非零解
        if beta_current > 0:
            # 尝试多个初始值来找到稳定解
            solutions_out = []
            for initial_guess in np.linspace(0, 1000, 1000):
                try:
                    x_solution = fsolve(steady_state_eq_out, initial_guess)[0]
                    if x_solution > 0:  # 只考虑正解
                        solutions_out.append(x_solution)
                except:
                    continue
            
            if solutions_out:
                # 选择最大的正解（对应稳定状态）
                x_solution_out = max(solutions_out)
                steady_state_solutions_out.append(x_solution_out)
                theoretical_results_out.append(x_solution_out)
            else:
                steady_state_solutions_out.append(0)
                theoretical_results_out.append(0)
        else:
            steady_state_solutions_out.append(0)
            theoretical_results_out.append(0)
        
        # 总贸易量是进口和出口之和
        total_solution = steady_state_solutions_in[-1] + steady_state_solutions_out[-1]
        steady_state_solutions.append(total_solution)
        theoretical_results.append(total_solution)
        
        if (step + 1) % 30 == 0:
            print(f"步骤 {step+1}/{len(removal_order_edges)} - 移除边: ({u}, {v}), 权重: {edge_weight:.4f}")
            print(f"alpha: {alpha_current:.4f}, beta: {beta_current:.4f}")
            print(f"进口解: {steady_state_solutions_in[-1]:.4f}, 出口解: {steady_state_solutions_out[-1]:.4f}, 总解: {total_solution:.4f}")
            print(f"移除比例 q: {q:.4f}")
    
    return q_values, theoretical_results, steady_state_solutions, steady_state_solutions_in, steady_state_solutions_out, alpha_values, beta_values

# 6. 定义改进的动力学方程函数 - 分开进口和出口，使用B_i，分母改为x_mean
def dynamics_import(t, x_in, B_vector, E_sub, node_indices, x_mean):
    """进口部分的动力学方程函数，用于solve_ivp，使用B_i，分母改为x_mean"""
    dxdt_in = np.zeros_like(x_in)
    n_nodes = len(x_in)
    
    for i in range(n_nodes):
        # 获取当前节点的B_i
        original_idx = node_indices[i]
        B_i = B_vector[original_idx]
        
        # 自身动力学
        dxdt_in[i] = B_i * x_in[i]
        
        # 进口交互作用项 - 分母改为x_mean
        import_interaction = 0
        for j in range(n_nodes):
            if i != j and E_sub[i, j] > 0:
                import_interaction += E_sub[i, j] * (x_in[j]**2 / (x_in[j]**2 + C1))
        
        dxdt_in[i] += import_interaction
    
    return dxdt_in

def dynamics_export(t, x_out, B_vector, G_sub, node_indices, x_mean):
    """出口部分的动力学方程函数，用于solve_ivp，使用B_i，分母改为x_mean"""
    dxdt_out = np.zeros_like(x_out)
    n_nodes = len(x_out)
    
    for i in range(n_nodes):
        # 获取当前节点的B_i
        original_idx = node_indices[i]
        B_i = B_vector[original_idx]
        
        # 自身动力学
        dxdt_out[i] = B_i * x_out[i]
        
        # 出口交互作用项 - 分母改为x_mean
        export_interaction = 0
        for j in range(n_nodes):
            if i != j and G_sub[i, j] > 0:
                export_interaction += G_sub[i, j] * (x_out[j]**2 / (x_out[j]**2 + C2))
        
        dxdt_out[i] += export_interaction
    
    return dxdt_out

# 7. 改进的韧性分析 - 基于动力学方程的仿真方法（分开进口和出口，使用B_i，分母改为x_mean，随机去边）
def simulate_dynamics_robustness_edge_removal(G_lcc, nodes, node_in_volumes, node_out_volumes, E, G_matrix, B_i_values, x_mean, removal_order_edges):
    """基于改进动力学方程的仿真韧性分析 - 分开进口和出口，使用B_i，分母改为x_mean，随机去边"""
    n = len(nodes)
    
    # 创建节点到索引的映射
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # 初始矩阵
    E_current = E.copy()
    G_current = G_matrix.copy()
    
    # 记录当前边状态（哪些边还在）
    current_edges = set(G_lcc.edges())
    
    # 记录每个q值对应的所有国家演化曲线（进口）
    all_country_steady_states_in = []
    all_country_trajectories_in = []
    
    # 记录每个q值对应的所有国家演化曲线（出口）
    all_country_steady_states_out = []
    all_country_trajectories_out = []
    
    # 记录每个q值对应的所有国家总贸易量演化曲线
    all_country_steady_states = []
    all_country_trajectories = []
    
    # 记录每个q值对应的网络平均进口量和平均出口量（从网络结构计算）
    simulation_alpha_values = []  # 网络平均进口量alpha（理论解析参数）
    simulation_beta_values = []   # 网络平均出口量beta（理论解析参数）
    
    # 记录每个q值对应的仿真稳态平均值
    simulation_avg_in_steady = []  # 仿真进口稳态平均值
    simulation_avg_out_steady = []  # 仿真出口稳态平均值
    simulation_avg_total_steady = []  # 仿真总贸易量稳态平均值
    
    # 记录第1步（移除第1条边时）的演化曲线
    step1_trajectories = None
    
    # 记录第10步（移除第10条边时）的演化曲线
    step10_trajectories = None
    
    # 数值积分参数
    t_span = (0, 100)  # 时间范围
    t_eval = np.linspace(0, 100, 100)  # 评估时间点
    steady_window = 10  # 用于计算稳态值的最后时间步数
    
    # 计算初始权重比例
    removed_weight = 0
    q_values = []
    
    # 使用进度条
    print("开始边移除仿真...")
    pbar = tqdm(total=len(removal_order_edges) + 1, desc="边移除进度")
    
    # 初始状态：没有边被移除
    q = 0.0
    q_values.append(q)
    
    # 计算当前网络的平均进口量和平均出口量（网络结构参数）
    # 创建临时图以计算当前加权度
    temp_G = nx.DiGraph()
    for node in nodes:
        temp_G.add_node(node)
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if E_current[i, j] > 0:
                temp_G.add_edge(node_j, node_i, weight=E_current[i, j])
            if G_current[i, j] > 0:
                temp_G.add_edge(node_i, node_j, weight=G_current[i, j])
    
    # 计算当前加权度
    current_in_volumes = []
    current_out_volumes = []
    for node in nodes:
        if node in temp_G.nodes():
            in_vol = sum([temp_G[u][node]['weight'] for u in temp_G.predecessors(node)])
            out_vol = sum([temp_G[node][v]['weight'] for v in temp_G.successors(node)])
        else:
            in_vol = 0
            out_vol = 0
        current_in_volumes.append(in_vol)
        current_out_volumes.append(out_vol)
    
    alpha_current = np.mean(current_in_volumes) if current_in_volumes else 0
    beta_current = np.mean(current_out_volumes) if current_out_volumes else 0
    
    simulation_alpha_values.append(alpha_current)
    simulation_beta_values.append(beta_current)
    
    # 初始状态（对数变换后的进口量和出口量）
    x0_in = np.array(current_in_volumes)  # 使用当前网络的加权度作为初始值
    x0_out = np.array(current_out_volumes)  # 使用当前网络的加权度作为初始值
    
    # 初始仿真：所有边都在
    try:
        # 求解进口动力学 - 分母改为x_mean
        sol_in = solve_ivp(
            dynamics_import, 
            t_span, 
            x0_in, 
            args=(B_i_values, E_current, list(range(n)), x_mean),
            t_eval=t_eval,
            method='RK45'
        )
        
        # 求解出口动力学 - 分母改为x_mean
        sol_out = solve_ivp(
            dynamics_export, 
            t_span, 
            x0_out, 
            args=(B_i_values, G_current, list(range(n)), x_mean),
            t_eval=t_eval,
            method='RK45'
        )
        
        if sol_in.success and sol_out.success:
            x_trajectory_in = sol_in.y.T  # 转置为(time, nodes)形状
            x_trajectory_out = sol_out.y.T
            
            # 计算总贸易量演化曲线
            x_trajectory_total = x_trajectory_in + x_trajectory_out
            
            # 取最后几步的平均值作为稳态值
            steady_states_in = np.mean(x_trajectory_in[-steady_window:], axis=0)
            steady_states_out = np.mean(x_trajectory_out[-steady_window:], axis=0)
            steady_states_total = steady_states_in + steady_states_out
            
            # 计算稳态平均值
            avg_in_steady = np.mean(steady_states_in)
            avg_out_steady = np.mean(steady_states_out)
            avg_total_steady = np.mean(steady_states_total)
            
            # 保存所有国家的稳态值
            all_country_steady_states_in.append(steady_states_in)
            all_country_steady_states_out.append(steady_states_out)
            all_country_steady_states.append(steady_states_total)
            
            # 保存演化曲线
            country_trajectories = x_trajectory_total
            all_country_trajectories.append(country_trajectories)
            
            # 保存仿真稳态平均值
            simulation_avg_in_steady.append(avg_in_steady)
            simulation_avg_out_steady.append(avg_out_steady)
            simulation_avg_total_steady.append(avg_total_steady)
            
        else:
            print('初始状态求解失败')
            # 使用初始值作为稳态值
            all_country_steady_states_in.append(x0_in)
            all_country_steady_states_out.append(x0_out)
            all_country_steady_states.append(x0_in + x0_out)
            country_trajectories = np.zeros((len(t_eval), n)) * np.nan
            all_country_trajectories.append(country_trajectories)
            
            # 保存仿真稳态平均值（使用初始值）
            simulation_avg_in_steady.append(np.mean(x0_in))
            simulation_avg_out_steady.append(np.mean(x0_out))
            simulation_avg_total_steady.append(np.mean(x0_in + x0_out))
    except:
        print('初始状态求解失败')
        # 使用初始值作为稳态值
        all_country_steady_states_in.append(x0_in)
        all_country_steady_states_out.append(x0_out)
        all_country_steady_states.append(x0_in + x0_out)
        country_trajectories = np.zeros((len(t_eval), n)) * np.nan
        all_country_trajectories.append(country_trajectories)
        
        # 保存仿真稳态平均值（使用初始值）
        simulation_avg_in_steady.append(np.mean(x0_in))
        simulation_avg_out_steady.append(np.mean(x0_out))
        simulation_avg_total_steady.append(np.mean(x0_in + x0_out))
    
    # 更新进度条
    pbar.update(1)
    
    # 按照随机顺序逐条边移除
    for step, (u, v, attr) in enumerate(removal_order_edges):
        if (u, v) not in current_edges:
            # 边已经被移除，跳过
            # 但仍然需要记录q值
            edge_weight = attr['weight']
            removed_weight += edge_weight
            q = removed_weight / total_edge_weight
            q_values.append(q)
            
            # 重复上一个状态的稳态值
            all_country_steady_states_in.append(all_country_steady_states_in[-1])
            all_country_steady_states_out.append(all_country_steady_states_out[-1])
            all_country_steady_states.append(all_country_steady_states[-1])
            all_country_trajectories.append(all_country_trajectories[-1])
            
            # 重复网络结构参数和仿真稳态平均值
            simulation_alpha_values.append(simulation_alpha_values[-1])
            simulation_beta_values.append(simulation_beta_values[-1])
            simulation_avg_in_steady.append(simulation_avg_in_steady[-1])
            simulation_avg_out_steady.append(simulation_avg_out_steady[-1])
            simulation_avg_total_steady.append(simulation_avg_total_steady[-1])
            
            pbar.update(1)
            continue
        
        # 获取边权重
        edge_weight = attr['weight']
        
        # 从当前边集合中移除该边
        current_edges.remove((u, v))
        
        # 更新矩阵
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        
        # 如果这是进口边（从v到u）
        if E_current[v_idx, u_idx] > 0:
            E_current[v_idx, u_idx] = 0
        
        # 如果这是出口边（从u到v）
        if G_current[u_idx, v_idx] > 0:
            G_current[u_idx, v_idx] = 0
        
        # 更新已移除的权重
        removed_weight += edge_weight
        
        # 计算当前移除率q（基于边权重）
        q = removed_weight / total_edge_weight
        q_values.append(q)
        
        # 更新动态初始值：重新计算当前网络的加权度
        # 创建临时图以计算当前加权度
        temp_G = nx.DiGraph()
        
        # 首先添加所有节点，确保它们存在
        for node in nodes:
            temp_G.add_node(node)
        
        # 然后添加边
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if E_current[i, j] > 0:
                    temp_G.add_edge(node_j, node_i, weight=E_current[i, j])
                if G_current[i, j] > 0:
                    temp_G.add_edge(node_i, node_j, weight=G_current[i, j])
        
        # 计算当前加权度 - 修复节点访问问题
        current_in_volumes = []
        current_out_volumes = []
        for node in nodes:
            # 检查节点是否在图中（应该都在，因为我们已经添加了所有节点）
            if node in temp_G.nodes():
                # 计算入度（进口量）
                in_vol = sum([temp_G[u][node]['weight'] for u in temp_G.predecessors(node)])
                # 计算出度（出口量）
                out_vol = sum([temp_G[node][v]['weight'] for v in temp_G.successors(node)])
            else:
                in_vol = 0
                out_vol = 0
            
            current_in_volumes.append(in_vol)
            current_out_volumes.append(out_vol)
        
        alpha_current = np.mean(current_in_volumes) if current_in_volumes else 0
        beta_current = np.mean(current_out_volumes) if current_out_volumes else 0
        
        simulation_alpha_values.append(alpha_current)
        simulation_beta_values.append(beta_current)
        
        # 更新初始值
        x0_in = np.array(current_in_volumes)
        x0_out = np.array(current_out_volumes)
        
        # 进行仿真
        try:
            # 求解进口动力学 - 分母改为x_mean
            sol_in = solve_ivp(
                dynamics_import, 
                t_span, 
                x0_in, 
                args=(B_i_values, E_current, list(range(n)), x_mean),
                t_eval=t_eval,
                method='RK45'
            )
            
            # 求解出口动力学 - 分母改为x_mean
            sol_out = solve_ivp(
                dynamics_export, 
                t_span, 
                x0_out, 
                args=(B_i_values, G_current, list(range(n)), x_mean),
                t_eval=t_eval,
                method='RK45'
            )
            
            if sol_in.success and sol_out.success:
                x_trajectory_in = sol_in.y.T
                x_trajectory_out = sol_out.y.T
                x_trajectory_total = x_trajectory_in + x_trajectory_out
                
                # 取最后几步的平均值作为稳态值
                steady_states_in = np.mean(x_trajectory_in[-steady_window:], axis=0)
                steady_states_out = np.mean(x_trajectory_out[-steady_window:], axis=0)
                steady_states_total = steady_states_in + steady_states_out
                
                # 计算稳态平均值
                avg_in_steady = np.mean(steady_states_in)
                avg_out_steady = np.mean(steady_states_out)
                avg_total_steady = np.mean(steady_states_total)
                
                # 保存所有国家的稳态值
                all_country_steady_states_in.append(steady_states_in)
                all_country_steady_states_out.append(steady_states_out)
                all_country_steady_states.append(steady_states_total)
                
                # 保存演化曲线
                country_trajectories = x_trajectory_total
                all_country_trajectories.append(country_trajectories)
                
                # 保存仿真稳态平均值
                simulation_avg_in_steady.append(avg_in_steady)
                simulation_avg_out_steady.append(avg_out_steady)
                simulation_avg_total_steady.append(avg_total_steady)
                
            else:
                print(f'步骤 {step+1} 求解失败')
                # 使用当前初始值作为稳态值
                all_country_steady_states_in.append(x0_in)
                all_country_steady_states_out.append(x0_out)
                all_country_steady_states.append(x0_in + x0_out)
                country_trajectories = np.zeros((len(t_eval), n)) * np.nan
                all_country_trajectories.append(country_trajectories)
                
                # 保存仿真稳态平均值（使用当前初始值）
                simulation_avg_in_steady.append(np.mean(x0_in))
                simulation_avg_out_steady.append(np.mean(x0_out))
                simulation_avg_total_steady.append(np.mean(x0_in + x0_out))
        except:
            print(f'步骤 {step+1} 求解失败')
            # 使用当前初始值作为稳态值
            all_country_steady_states_in.append(x0_in)
            all_country_steady_states_out.append(x0_out)
            all_country_steady_states.append(x0_in + x0_out)
            country_trajectories = np.zeros((len(t_eval), n)) * np.nan
            all_country_trajectories.append(country_trajectories)
            
            # 保存仿真稳态平均值（使用当前初始值）
            simulation_avg_in_steady.append(np.mean(x0_in))
            simulation_avg_out_steady.append(np.mean(x0_out))
            simulation_avg_total_steady.append(np.mean(x0_in + x0_out))
        
        # 记录第1步（移除第1条边时）的演化曲线
        if step == 500:
            step1_trajectories = country_trajectories
        
        # 记录第10步（移除第10条边时）的演化曲线
        if step == 600:
            step10_trajectories = country_trajectories
        
        # 更新进度条
        pbar.update(1)
    
    pbar.close()
    
    return (q_values, simulation_avg_total_steady, simulation_avg_in_steady, simulation_avg_out_steady,
            all_country_steady_states, all_country_steady_states_in, all_country_steady_states_out,
            all_country_trajectories, step1_trajectories, step10_trajectories,
            simulation_alpha_values, simulation_beta_values)

# 8. 执行韧性分析
print("开始改进的理论解析韧性分析（随机去边）...")
theoretical_q_values, theoretical_results_total, steady_state_solutions_total, steady_state_solutions_in, steady_state_solutions_out, alpha_values, beta_values = theoretical_robustness_analysis_edge_removal(
    G_lcc, nodes, node_in_volumes, node_out_volumes, B_solution, x_mean, random_removal_order_edges)

print("开始基于改进动力学方程的仿真韧性分析（随机去边）...")
simulation_q_values, simulation_avg_total_steady, simulation_avg_in_steady, simulation_avg_out_steady, all_country_steady_states_total, all_country_steady_states_in, all_country_steady_states_out, all_country_trajectories_total, step1_trajectories, step10_trajectories, simulation_alpha_values, simulation_beta_values = simulate_dynamics_robustness_edge_removal(
    G_lcc, nodes, node_in_volumes, node_out_volumes, E, G_matrix, B_i_values, x_mean, random_removal_order_edges)

# 9. 保存结果
# 保存韧性分析结果
results_df = pd.DataFrame({
    'Removal_Ratio': simulation_q_values,
    'Theoretical_Resilience_Total': np.interp(simulation_q_values, theoretical_q_values, theoretical_results_total),
    'Theoretical_Resilience_In': np.interp(simulation_q_values, theoretical_q_values, steady_state_solutions_in),
    'Theoretical_Resilience_Out': np.interp(simulation_q_values, theoretical_q_values, steady_state_solutions_out),
    'Simulation_Resilience_Total': simulation_avg_total_steady,
    'Simulation_Resilience_In': simulation_avg_in_steady,
    'Simulation_Resilience_Out': simulation_avg_out_steady,
    'Alpha': simulation_alpha_values,
    'Beta': simulation_beta_values
})

results_file = os.path.join(output_dir, "resilience_analysis_results_edge_removal.csv")
results_df.to_csv(results_file, index=False)
print(f"韧性分析结果已保存至: {results_file}")

# 保存每个q值对应的解
solutions_df = pd.DataFrame({
    'Removal_Ratio': theoretical_q_values,
    'Steady_State_Solution_Total': theoretical_results_total,
    'Steady_State_Solution_In': steady_state_solutions_in,
    'Steady_State_Solution_Out': steady_state_solutions_out,
    'Alpha': alpha_values,
    'Beta': beta_values
})

solutions_file = os.path.join(output_dir, "steady_state_solutions_edge_removal.csv")
solutions_df.to_csv(solutions_file, index=False)
print(f"稳态解已保存至: {solutions_file}")

# 保存每个国家随q的变化曲线
country_curves_total_df = pd.DataFrame(all_country_steady_states_total, index=simulation_q_values)
country_curves_total_df.columns = nodes
country_curves_file = os.path.join(output_dir, "country_curves_vs_q_edge_removal.csv")
country_curves_total_df.to_csv(country_curves_file)
print(f"国家随q变化曲线已保存至: {country_curves_file}")

# 保存第1步（移除第1条边时）所有国家的演化曲线
if step1_trajectories is not None:
    step1_trajectories_df = pd.DataFrame(step1_trajectories, columns=nodes)
    step1_trajectories_file = os.path.join(output_dir, "country_trajectories_step1_edge_removal.csv")
    step1_trajectories_df.to_csv(step1_trajectories_file, index=False)
    print(f"第1步（移除第1条边时）国家演化曲线已保存至: {step1_trajectories_file}")

# 保存第10步（移除第10条边时）所有国家的演化曲线
if step10_trajectories is not None:
    step10_trajectories_df = pd.DataFrame(step10_trajectories, columns=nodes)
    step10_trajectories_file = os.path.join(output_dir, "country_trajectories_step10_edge_removal.csv")
    step10_trajectories_df.to_csv(step10_trajectories_file, index=False)
    print(f"第10步（移除第10条边时）国家演化曲线已保存至: {step10_trajectories_file}")

# ================ 绘图部分 ================
macaron_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

# 1. 第一张图：网络总的稳态值<x>随权重移除比例q_e的变化曲线
print("\n绘制第一张图：网络总的稳态值<x>随q_e的变化曲线...")

plt.figure(figsize=(14, 10))

# 绘制理论曲线（总贸易量）
theoretical_plot_values_total = [max(val, 1e-5) for val in theoretical_results_total]
plt.plot(theoretical_q_values, theoretical_plot_values_total, 
         label='Theory-based ODE', color=macaron_colors[1], linewidth=8, linestyle='-')

# 绘制仿真点（总贸易量）
plt.scatter(simulation_q_values, [max(val, 1e-5) for val in simulation_avg_total_steady],
            label='Simulation-based HDE', color=macaron_colors[0],
            s=100, marker='^', alpha=0.7, zorder=5)

plt.axhline(y=np.max(theoretical_results_total),
            color='gray',      # 颜色随意
            linestyle='--',    # 虚线
            linewidth=3)  

# 检测阈值点 - 总稳态随q_e的变化
threshold_q_total = None
threshold_y_total = None

if len(theoretical_q_values) > 1 and len(theoretical_results_total) > 1:
    # 寻找稳态值从大于0到0变化或从0到大于0变化的临界点
    threshold_tol = 1e-3
    
    # 首先找到所有稳态值接近0的点
    zero_indices = []
    for i, val in enumerate(theoretical_results_total):
        if abs(val) < threshold_tol:
            zero_indices.append(i)
    
    # 如果没有直接为0的点，找第一个小于阈值的点
    if zero_indices:
        # 找第一个从非零到零变化的点
        for i in range(1, len(theoretical_results_total)):
            if theoretical_results_total[i-1] > threshold_tol and abs(theoretical_results_total[i]) < threshold_tol:
                threshold_q_total = theoretical_q_values[i]
                threshold_y_total = theoretical_results_total[i]
                break
            elif abs(theoretical_results_total[i-1]) < threshold_tol and theoretical_results_total[i] > threshold_tol:
                threshold_q_total = theoretical_q_values[i]
                threshold_y_total = theoretical_results_total[i]
                break
    else:
        # 找第一个小于阈值的点
        for i, val in enumerate(theoretical_results_total):
            if val < threshold_tol:
                threshold_q_total = theoretical_q_values[i]
                threshold_y_total = val
                break

# 标注阈值点
if threshold_q_total is not None:
    # 标记阈值点
    plt.plot(threshold_q_total, max(threshold_y_total, 1e-5), 'ro', markersize=12, markeredgecolor='red', 
             markeredgewidth=2, zorder=10)
    
    # 添加文本标注
    annotation_text = f'$q_e^{{*}}={threshold_q_total:.3f}$'
    
    # 确定标注位置
    if threshold_q_total > 0.5:
        text_x = threshold_q_total - 0.08
    else:
        text_x = threshold_q_total + 0.08
    
    y_pos = max(threshold_y_total, 1e-5) * 0.1
    if y_pos < 0.1:
        y_pos = 0.1
    
    plt.annotate(annotation_text, 
                 xy=(threshold_q_total - 0.01, max(threshold_y_total, 1e-5)),
                 xytext=(text_x - 0.16, y_pos *0.5),
                 fontsize=font_size-8,
                 color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5, connectionstyle="arc3,rad=0"),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='red', alpha=0.5),
                 zorder=10)

# 设置y轴为symlog刻度
plt.yscale('symlog', linthresh=0.1)

plt.xlabel(r'$q_e$', fontsize=font_size-3)
plt.ylabel(r'$\langle x \rangle$', fontsize=font_size-3)
plt.xticks(fontsize=font_size-5)
plt.yticks(fontsize=font_size-5)
leg = plt.legend(fontsize=font_size-5, loc='center left', bbox_to_anchor=(0.08, 0.5),
           labelspacing=0.9,   # <-- 行间距系数，默认 0.5
           edgecolor='black')
leg.get_frame().set_alpha(0.7)        # 0=全透明，1=不透明
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(-0.03, 1000)
plt.tight_layout()

# 保存图片
figure_1_file = os.path.join(figure_dir, "01_total_steady_vs_qe.png")
plt.savefig(figure_1_file, dpi=600, bbox_inches='tight')
print(f"第一张图已保存至: {figure_1_file}")
plt.show()

# 2. 第二张图：进口和出口稳态随q_e的变化曲线
print("\n绘制第二张图：进口和出口稳态随q_e的变化曲线...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# 左侧子图：进口稳态<x_in>随q_e的变化
theoretical_plot_values_in = [max(val, 1e-5) for val in steady_state_solutions_in]
ax1.plot(theoretical_q_values, theoretical_plot_values_in, 
         label='Theory-based ODE', color=macaron_colors[1], linewidth=5, linestyle='-')

ax1.scatter(simulation_q_values, [max(val, 1e-5) for val in simulation_avg_in_steady],
            label='Simulation-based HDE', color=macaron_colors[0],
            s=120, marker='^', alpha=0.8, zorder=5)

ax1.axhline(y=np.max(theoretical_plot_values_in),
            color='gray',      # 颜色随意
            linestyle='--',    # 虚线
            linewidth=3)  

# 检测进口阈值点
threshold_q_in = None
threshold_y_in = None

if len(theoretical_q_values) > 1 and len(steady_state_solutions_in) > 1:
    threshold_tol = 1e-3
    
    zero_indices = []
    for i, val in enumerate(steady_state_solutions_in):
        if abs(val) < threshold_tol:
            zero_indices.append(i)
    
    if zero_indices:
        for i in range(1, len(steady_state_solutions_in)):
            if steady_state_solutions_in[i-1] > threshold_tol and abs(steady_state_solutions_in[i]) < threshold_tol:
                threshold_q_in = theoretical_q_values[i]
                threshold_y_in = steady_state_solutions_in[i]
                break
            elif abs(steady_state_solutions_in[i-1]) < threshold_tol and steady_state_solutions_in[i] > threshold_tol:
                threshold_q_in = theoretical_q_values[i]
                threshold_y_in = steady_state_solutions_in[i]
                break
    else:
        for i, val in enumerate(steady_state_solutions_in):
            if val < threshold_tol:
                threshold_q_in = theoretical_q_values[i]
                threshold_y_in = val
                break

# 标注进口阈值点
if threshold_q_in is not None:
    ax1.plot(threshold_q_in, max(threshold_y_in, 1e-5), 'ro', markersize=12, markeredgecolor='red', 
             markeredgewidth=2, zorder=10)
    
    annotation_text = f'$q_{{e,in}}^{{*}}={threshold_q_in:.3f}$'
    
    if threshold_q_in > 0.5:
        text_x = threshold_q_in - 0.08
    else:
        text_x = threshold_q_in + 0.08
    
    y_pos = max(threshold_y_in, 1e-5) * 0.5
    if y_pos < 0.1:
        y_pos = 0.1
    
    ax1.annotate(annotation_text, 
                 xy=(threshold_q_in - 0.01, max(threshold_y_in, 1e-5)),
                 xytext=(text_x - 0.3, y_pos * 0.4),
                 fontsize=font_size-8,
                 color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5, connectionstyle="arc3,rad=0"),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='red', alpha=0.5),
                 zorder=10)

ax1.set_yscale('symlog', linthresh=0.1)
ax1.set_xlabel(r'$q_e$', fontsize=font_size-3)
ax1.set_ylabel(r'$\langle x_{in} \rangle$', fontsize=font_size-3)
ax1.set_xticks(np.arange(0, 1.1, 0.2))
ax1.set_xlim(0, 1)
ax1.tick_params(labelsize=font_size-8)
ax1.legend(fontsize=font_size-8, loc='center right', bbox_to_anchor=(0.75, 0.45))
ax1.grid(True, alpha=0.3)

# 右侧子图：出口稳态<x_out>随q_e的变化
theoretical_plot_values_out = [max(val, 1e-5) for val in steady_state_solutions_out]
ax2.plot(theoretical_q_values, theoretical_plot_values_out, 
         label='Theory-based ODE', color=macaron_colors[1], linewidth=5, linestyle='-')

ax2.scatter(simulation_q_values, [max(val, 1e-5) for val in simulation_avg_out_steady],
            label='Simulation-based HDE', color=macaron_colors[0],
            s=120, marker='^', alpha=0.8, zorder=5)

ax2.axhline(y=np.max(theoretical_plot_values_out),
            color='gray',      # 颜色随意
            linestyle='--',    # 虚线
            linewidth=3)  

# 检测出口阈值点
threshold_q_out = None
threshold_y_out = None

if len(theoretical_q_values) > 1 and len(steady_state_solutions_out) > 1:
    threshold_tol = 1e-3
    
    zero_indices = []
    for i, val in enumerate(steady_state_solutions_out):
        if abs(val) < threshold_tol:
            zero_indices.append(i)
    
    if zero_indices:
        for i in range(1, len(steady_state_solutions_out)):
            if steady_state_solutions_out[i-1] > threshold_tol and abs(steady_state_solutions_out[i]) < threshold_tol:
                threshold_q_out = theoretical_q_values[i]
                threshold_y_out = steady_state_solutions_out[i]
                break
            elif abs(steady_state_solutions_out[i-1]) < threshold_tol and steady_state_solutions_out[i] > threshold_tol:
                threshold_q_out = theoretical_q_values[i]
                threshold_y_out = steady_state_solutions_out[i]
                break
    else:
        for i, val in enumerate(steady_state_solutions_out):
            if val < threshold_tol:
                threshold_q_out = theoretical_q_values[i]
                threshold_y_out = val
                break

# 标注出口阈值点
if threshold_q_out is not None:
    ax2.plot(threshold_q_out, max(threshold_y_out, 1e-5), 'ro', markersize=12, markeredgecolor='red', 
             markeredgewidth=2, zorder=10)
    
    annotation_text = f'$q_{{e,out}}^{{*}}={threshold_q_out:.3f}$'
    
    if threshold_q_out > 0.5:
        text_x = threshold_q_out - 0.08
    else:
        text_x = threshold_q_out + 0.08
    
    y_pos = max(threshold_y_out, 1e-5) * 0.5
    if y_pos < 0.1:
        y_pos = 0.1
    
    ax2.annotate(annotation_text, 
                 xy=(threshold_q_out - 0.01, max(threshold_y_out, 1e-5)),
                 xytext=(text_x - 0.3, y_pos * 0.4),
                 fontsize=font_size-8,
                 color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5, connectionstyle="arc3,rad=0"),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='red', alpha=0.5),
                 zorder=10)

ax2.set_yscale('symlog', linthresh=0.1)
ax2.set_xlabel(r'$q_e$', fontsize=font_size-3)
ax2.set_ylabel(r'$\langle x_{out} \rangle$', fontsize=font_size-3)
ax2.set_xticks(np.arange(0, 1.1, 0.2))
ax2.set_xlim(0, 1)
ax2.tick_params(labelsize=font_size-8)
ax2.legend(fontsize=font_size-8, loc='center right', bbox_to_anchor=(0.75, 0.45))
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图片
figure_2_file = os.path.join(figure_dir, "02_import_export_steady_vs_qe.png")
plt.savefig(figure_2_file, dpi=600, bbox_inches='tight')
print(f"第二张图已保存至: {figure_2_file}")
plt.show()

# 3. 第三张图：总稳态<x>随alpha_eff和beta_eff的变化
print("\n绘制第三张图：总稳态<x>随alpha_eff和beta_eff的变化...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# 左侧子图：总稳态<x>随alpha_eff的变化
theoretical_curve_total_alpha_x = np.array(alpha_values)
theoretical_curve_total_alpha_y = np.array(theoretical_results_total)  # 总稳态值

ax1.plot(theoretical_curve_total_alpha_x, [max(val, 1e-5) for val in theoretical_curve_total_alpha_y], 
         label='Theory-based ODE', color=macaron_colors[1], linewidth=5)

# 绘制仿真分析的点
if len(simulation_alpha_values) == len(simulation_avg_total_steady):
    ax1.scatter(simulation_alpha_values, [max(val, 1e-5) for val in simulation_avg_total_steady],
                label='Simulation-based HDE', color=macaron_colors[0], s=100, marker='^', alpha=0.8, zorder=5)

ax1.axhline(y=np.max(theoretical_curve_total_alpha_y),
            color='gray',      # 颜色随意
            linestyle='--',    # 虚线
            linewidth=3)  

# 检测阈值点 - 总稳态随alpha_eff的变化
threshold_alpha_total = None
threshold_alpha_y_total = None

if len(theoretical_curve_total_alpha_x) > 1 and len(theoretical_curve_total_alpha_y) > 1:
    threshold_tol = 1e-3
    
    zero_indices = []
    for i, val in enumerate(theoretical_curve_total_alpha_y):
        if abs(val) < threshold_tol:
            zero_indices.append(i)
    
    if zero_indices:
        for i in range(1, len(theoretical_curve_total_alpha_y)):
            if theoretical_curve_total_alpha_y[i-1] > threshold_tol and abs(theoretical_curve_total_alpha_y[i]) < threshold_tol:
                threshold_alpha_total = theoretical_curve_total_alpha_x[i]
                threshold_alpha_y_total = theoretical_curve_total_alpha_y[i]
                break
            elif abs(theoretical_curve_total_alpha_y[i-1]) < threshold_tol and theoretical_curve_total_alpha_y[i] > threshold_tol:
                threshold_alpha_total = theoretical_curve_total_alpha_x[i]
                threshold_alpha_y_total = theoretical_curve_total_alpha_y[i]
                break
    else:
        for i, val in enumerate(theoretical_curve_total_alpha_y):
            if val < threshold_tol:
                threshold_alpha_total = theoretical_curve_total_alpha_x[i]
                threshold_alpha_y_total = val
                break

# 标注阈值点
if threshold_alpha_total is not None:
    ax1.plot(threshold_alpha_total, max(threshold_alpha_y_total, 1e-5), 'ro', markersize=12, 
             markeredgecolor='red', markeredgewidth=2, zorder=10)
    
    annotation_text = f'$\\alpha_{{eff,total}}^{{*}}={threshold_alpha_total:.3f}$'
    
    xlim = ax1.get_xlim()
    x_range = xlim[1] - xlim[0]
    
    if threshold_alpha_total > (xlim[0] + x_range * 0.6):
        text_x = threshold_alpha_total - x_range * 0.1
    else:
        text_x = threshold_alpha_total + x_range * 0.1
    
    y_pos = max(threshold_alpha_y_total, 1e-5) * 0.5
    if y_pos < 0.1:
        y_pos = 0.1
    
    ax1.annotate(annotation_text, 
                 xy=(threshold_alpha_total + 1, max(threshold_alpha_y_total, 1e-5)),
                 xytext=(text_x, y_pos * 0.4),
                 fontsize=font_size-10,
                 color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=2, connectionstyle="arc3,rad=0"),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='red', alpha=0.6),
                 zorder=10)

ax1.set_yscale('symlog', linthresh=0.1)
ax1.set_xlabel(r'$\alpha_{eff}$', fontsize=font_size-3)
ax1.set_ylabel(r'$\langle x \rangle$', fontsize=font_size-3)
ax1.legend(fontsize=font_size-5, loc='center right', bbox_to_anchor=(0.98, 0.5))
ax1.set_ylim([-0.01, 10e2])
ax1.set_xlim([0, 100])
ax1.tick_params(labelsize=font_size-5)
ax1.grid(True, alpha=0.3)

# 右侧子图：总稳态<x>随beta_eff的变化
theoretical_curve_total_beta_x = np.array(beta_values)
theoretical_curve_total_beta_y = np.array(theoretical_results_total)  # 总稳态值

ax2.plot(theoretical_curve_total_beta_x, [max(val, 1e-5) for val in theoretical_curve_total_beta_y], 
         label='Theory-based ODE', color=macaron_colors[1], linewidth=5)

# 绘制仿真分析的点
if len(simulation_beta_values) == len(simulation_avg_total_steady):
    ax2.scatter(simulation_beta_values, [max(val, 1e-5) for val in simulation_avg_total_steady],
                label='Simulation-based HDE', color=macaron_colors[0], s=100, marker='^', alpha=0.8, zorder=5)

ax2.axhline(y=np.max(theoretical_curve_total_beta_y),
            color='gray',      # 颜色随意
            linestyle='--',    # 虚线
            linewidth=3)  

# 检测阈值点 - 总稳态随beta_eff的变化
threshold_beta_total = None
threshold_beta_y_total = None

if len(theoretical_curve_total_beta_x) > 1 and len(theoretical_curve_total_beta_y) > 1:
    threshold_tol = 1e-3
    
    zero_indices = []
    for i, val in enumerate(theoretical_curve_total_beta_y):
        if abs(val) < threshold_tol:
            zero_indices.append(i)
    
    if zero_indices:
        for i in range(1, len(theoretical_curve_total_beta_y)):
            if theoretical_curve_total_beta_y[i-1] > threshold_tol and abs(theoretical_curve_total_beta_y[i]) < threshold_tol:
                threshold_beta_total = theoretical_curve_total_beta_x[i]
                threshold_beta_y_total = theoretical_curve_total_beta_y[i]
                break
            elif abs(theoretical_curve_total_beta_y[i-1]) < threshold_tol and theoretical_curve_total_beta_y[i] > threshold_tol:
                threshold_beta_total = theoretical_curve_total_beta_x[i]
                threshold_beta_y_total = theoretical_curve_total_beta_y[i]
                break
    else:
        for i, val in enumerate(theoretical_curve_total_beta_y):
            if val < threshold_tol:
                threshold_beta_total = theoretical_curve_total_beta_x[i]
                threshold_beta_y_total = val
                break

# 标注阈值点
if threshold_beta_total is not None:
    ax2.plot(threshold_beta_total, max(threshold_beta_y_total, 1e-5), 'ro', markersize=12, 
             markeredgecolor='red', markeredgewidth=2, zorder=10)
    
    annotation_text = f'$\\beta_{{eff,total}}^{{*}}={threshold_beta_total:.3f}$'
    
    xlim = ax2.get_xlim()
    x_range = xlim[1] - xlim[0]
    
    if threshold_beta_total > (xlim[0] + x_range * 0.6):
        text_x = threshold_beta_total - x_range * 0.1
    else:
        text_x = threshold_beta_total + x_range * 0.1
    
    y_pos = max(threshold_beta_y_total, 1e-5) * 0.5
    if y_pos < 0.1:
        y_pos = 0.1
    
    ax2.annotate(annotation_text, 
                 xy=(threshold_beta_total + 1, max(threshold_beta_y_total, 1e-5)),
                 xytext=(text_x, y_pos * 0.4),
                 fontsize=font_size-10,
                 color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=2, connectionstyle="arc3,rad=0"),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='red', alpha=0.8),
                 zorder=10)

ax2.set_yscale('symlog', linthresh=0.1)
ax2.set_xlabel(r'$\beta_{eff}$', fontsize=font_size-3)
ax2.set_ylabel(r'$\langle x \rangle$', fontsize=font_size-3)
ax2.legend(fontsize=font_size-5, loc='center right', bbox_to_anchor=(0.98, 0.5))
ax2.grid(True, alpha=0.3)
ax2.set_ylim([-0.01, 10e2])
ax2.set_xlim([0, 100])
ax2.tick_params(labelsize=font_size-5)

plt.tight_layout()

# 保存图片
figure_3_file = os.path.join(figure_dir, "03_total_steady_vs_alpha_beta.png")
plt.savefig(figure_3_file, dpi=600, bbox_inches='tight')
print(f"第三张图已保存至: {figure_3_file}") 
plt.show()

# 4. 第四张图：进口稳态<x_in> vs alpha_eff 和出口稳态<x_out> vs beta_eff
print("\n绘制第四张图：进口稳态<x_in> vs alpha_eff 和出口稳态<x_out> vs beta_eff...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# 左侧子图：进口稳态<x_in> vs alpha_eff
theoretical_curve_alpha_x = np.array(alpha_values)
theoretical_curve_alpha_y = np.array(steady_state_solutions_in)

ax1.plot(theoretical_curve_alpha_x, [max(val, 1e-5) for val in theoretical_curve_alpha_y], 
         label='Theory-based ODE', color=macaron_colors[1], linewidth=5)

# 绘制仿真分析的点
if len(simulation_alpha_values) == len(simulation_avg_in_steady):
    ax1.scatter(simulation_alpha_values, [max(val, 1e-5) for val in simulation_avg_in_steady],
                label='Simulation-based HDE', color=macaron_colors[0], s=100, marker='^', alpha=0.8, zorder=5)

ax1.axhline(y=np.max(theoretical_curve_alpha_y),
            color='gray',      # 颜色随意
            linestyle='--',    # 虚线
            linewidth=3)  

# 检测阈值点 - 进口部分
threshold_alpha_in = None
threshold_alpha_y_in = None

if len(theoretical_curve_alpha_x) > 1 and len(theoretical_curve_alpha_y) > 1:
    threshold_tol = 1e-3
    
    zero_indices = []
    for i, val in enumerate(theoretical_curve_alpha_y):
        if abs(val) < threshold_tol:
            zero_indices.append(i)
    
    if zero_indices:
        for i in range(1, len(theoretical_curve_alpha_y)):
            if theoretical_curve_alpha_y[i-1] > threshold_tol and abs(theoretical_curve_alpha_y[i]) < threshold_tol:
                threshold_alpha_in = theoretical_curve_alpha_x[i]
                threshold_alpha_y_in = theoretical_curve_alpha_y[i]
                break
            elif abs(theoretical_curve_alpha_y[i-1]) < threshold_tol and theoretical_curve_alpha_y[i] > threshold_tol:
                threshold_alpha_in = theoretical_curve_alpha_x[i]
                threshold_alpha_y_in = theoretical_curve_alpha_y[i]
                break
    else:
        for i, val in enumerate(theoretical_curve_alpha_y):
            if val < threshold_tol:
                threshold_alpha_in = theoretical_curve_alpha_x[i]
                threshold_alpha_y_in = val
                break

# 标注阈值点 - 进口部分
if threshold_alpha_in is not None:
    ax1.plot(threshold_alpha_in, max(threshold_alpha_y_in, 1e-5), 'ro', markersize=12, 
             markeredgecolor='red', markeredgewidth=2, zorder=10)
    
    annotation_text = f'$\\alpha_{{eff,in}}^{{*}}={threshold_alpha_in:.3f}$'
    
    xlim = ax1.get_xlim()
    x_range = xlim[1] - xlim[0]
    
    if threshold_alpha_in > (xlim[0] + x_range * 0.6):
        text_x = threshold_alpha_in - x_range * 0.1
    else:
        text_x = threshold_alpha_in + x_range * 0.1
    
    y_pos = max(threshold_alpha_y_in, 1e-5) * 0.5
    if y_pos < 0.1:
        y_pos = 0.1
    
    ax1.annotate(annotation_text, 
                 xy=(threshold_alpha_in + 1, max(threshold_alpha_y_in, 1e-5)),
                 xytext=(text_x, y_pos * 0.4),
                 fontsize=font_size-10,
                 color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=2, connectionstyle="arc3,rad=0"),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='red', alpha=0.6),
                 zorder=10)

ax1.set_yscale('symlog', linthresh=0.1)
ax1.set_xlabel(r'$\alpha_{eff}$', fontsize=font_size-3)
ax1.set_ylabel(r'$\langle x_{in} \rangle$', fontsize=font_size-3)
ax1.legend(fontsize=font_size-5, loc='center right', bbox_to_anchor=(0.98, 0.4))
ax1.set_ylim([-0.02, 10e2])
ax1.set_xlim([-2, 100])
ax1.tick_params(labelsize=font_size-5)
ax1.grid(True, alpha=0.3)

# 右侧子图：出口稳态<x_out> vs beta_eff
theoretical_curve_beta_x = np.array(beta_values)
theoretical_curve_beta_y = np.array(steady_state_solutions_out)

ax2.plot(theoretical_curve_beta_x, [max(val, 1e-5) for val in theoretical_curve_beta_y], 
         label='Theory-based ODE', color=macaron_colors[1], linewidth=5)

# 绘制仿真分析的点
if len(simulation_beta_values) == len(simulation_avg_out_steady):
    ax2.scatter(simulation_beta_values, [max(val, 1e-5) for val in simulation_avg_out_steady],
                label='Simulation-based HDE', color=macaron_colors[0], s=100, marker='^', alpha=0.8, zorder=5)

ax2.axhline(y=np.max(theoretical_curve_beta_y),
            color='gray',      # 颜色随意
            linestyle='--',    # 虚线
            linewidth=3)  

# 检测阈值点 - 出口部分
threshold_beta_out = None
threshold_beta_y_out = None

if len(theoretical_curve_beta_x) > 1 and len(theoretical_curve_beta_y) > 1:
    threshold_tol = 1e-3
    
    zero_indices = []
    for i, val in enumerate(theoretical_curve_beta_y):
        if abs(val) < threshold_tol:
            zero_indices.append(i)
    
    if zero_indices:
        for i in range(1, len(theoretical_curve_beta_y)):
            if theoretical_curve_beta_y[i-1] > threshold_tol and abs(theoretical_curve_beta_y[i]) < threshold_tol:
                threshold_beta_out = theoretical_curve_beta_x[i]
                threshold_beta_y_out = theoretical_curve_beta_y[i]
                break
            elif abs(theoretical_curve_beta_y[i-1]) < threshold_tol and theoretical_curve_beta_y[i] > threshold_tol:
                threshold_beta_out = theoretical_curve_beta_x[i]
                threshold_beta_y_out = theoretical_curve_beta_y[i]
                break
    else:
        for i, val in enumerate(theoretical_curve_beta_y):
            if val < threshold_tol:
                threshold_beta_out = theoretical_curve_beta_x[i]
                threshold_beta_y_out = val
                break

# 标注阈值点 - 出口部分
if threshold_beta_out is not None:
    ax2.plot(threshold_beta_out, max(threshold_beta_y_out, 1e-5), 'ro', markersize=12, 
             markeredgecolor='red', markeredgewidth=2, zorder=10)
    
    annotation_text = f'$\\beta_{{eff,out}}^{{*}}={threshold_beta_out:.3f}$'
    
    xlim = ax2.get_xlim()
    x_range = xlim[1] - xlim[0]
    
    if threshold_beta_out > (xlim[0] + x_range * 0.6):
        text_x = threshold_beta_out - x_range * 0.1
    else:
        text_x = threshold_beta_out + x_range * 0.1
    
    y_pos = max(threshold_beta_y_out, 1e-5) * 0.5
    if y_pos < 0.1:
        y_pos = 0.1
    
    ax2.annotate(annotation_text, 
                 xy=(threshold_beta_out + 1, max(threshold_beta_y_out, 1e-5)),
                 xytext=(text_x, y_pos * 0.4),
                 fontsize=font_size-10,
                 color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=2, connectionstyle="arc3,rad=0"),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='red', alpha=0.8),
                 zorder=10)

ax2.set_yscale('symlog', linthresh=0.1)
ax2.set_xlabel(r'$\beta_{eff}$', fontsize=font_size-3)
ax2.set_ylabel(r'$\langle x_{out} \rangle$', fontsize=font_size-3)
ax2.legend(fontsize=font_size-5, loc='center right', bbox_to_anchor=(0.98, 0.4))
ax2.grid(True, alpha=0.3)
ax2.set_ylim([-0.02, 10e2])
ax2.set_xlim([-2, 100])
ax2.tick_params(labelsize=font_size-5)

plt.tight_layout()

# 保存图片
figure_4_file = os.path.join(figure_dir, "04_import_vs_alpha_export_vs_beta.png")
plt.savefig(figure_4_file, dpi=600, bbox_inches='tight')
print(f"第四张图已保存至: {figure_4_file}")
plt.show()

# 5. 第五张图：第1步（移除第1条边时）的演化曲线
print("\n绘制第五张图：第1步（移除第1条边时）的演化曲线...")

if step1_trajectories is not None:
    plt.figure(figsize=(14, 10))
    
    # 时间轴
    time_axis = np.linspace(0, 100, step1_trajectories.shape[0])
    
    # 绘制每个国家的演化曲线（浅色细线）
    for i, node in enumerate(nodes):
        if not np.all(np.isnan(step1_trajectories[:, i])):
            trajectory = step1_trajectories[:, i]
            # 避免0值
            trajectory_plot = np.where(trajectory <= 0, 1e-5, trajectory)
            plt.plot(time_axis, trajectory_plot, 
                    color=macaron_colors[5], alpha=0.6, linewidth=1)
    
    # 计算并绘制平均演化曲线
    avg_trajectory = np.nanmean(step1_trajectories, axis=1)
    # 避免0值
    avg_trajectory_plot = np.where(avg_trajectory <= 0, 1e-5, avg_trajectory)
    plt.plot(time_axis, avg_trajectory_plot, 
             label='Average', color=macaron_colors[2], linewidth=3)
    
    # 设置y轴为symlog刻度
    plt.yscale('symlog', linthresh=0.1)
    
    plt.xlabel('Time', fontsize=font_size-3)
    plt.ylabel(r'$\langle x \rangle$', fontsize=font_size-3)
    plt.xticks(fontsize=font_size-8)
    plt.yticks(fontsize=font_size-8)
    plt.legend(fontsize=font_size-8, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    figure_5_file = os.path.join(figure_dir, "05_trajectories_step1.png")
    plt.savefig(figure_5_file, dpi=300, bbox_inches='tight')
    print(f"第五张图已保存至: {figure_5_file}")
    plt.show()

# 6. 第六张图：第10步（移除第10条边时）的演化曲线
print("\n绘制第六张图：第10步（移除第10条边时）的演化曲线...")

if step10_trajectories is not None:
    plt.figure(figsize=(14, 10))
    
    # 时间轴
    time_axis = np.linspace(0, 100, step10_trajectories.shape[0])
    
    # 绘制每个国家的演化曲线（浅色细线）
    for i, node in enumerate(nodes):
        if not np.all(np.isnan(step10_trajectories[:, i])):
            trajectory = step10_trajectories[:, i]
            # 避免0值
            trajectory_plot = np.where(trajectory <= 0, 1e-5, trajectory)
            plt.plot(time_axis, trajectory_plot, 
                    color=macaron_colors[5], alpha=0.6, linewidth=1)
    
    # 计算并绘制平均演化曲线
    avg_trajectory = np.nanmean(step10_trajectories, axis=1)
    # 避免0值
    avg_trajectory_plot = np.where(avg_trajectory <= 0, 1e-5, avg_trajectory)
    plt.plot(time_axis, avg_trajectory_plot, 
             label='Average', color=macaron_colors[3], linewidth=3)
    
    # 设置y轴为symlog刻度
    plt.yscale('symlog', linthresh=0.1)
    
    plt.xlabel('Time', fontsize=font_size-3)
    plt.ylabel(r'$\langle x \rangle$', fontsize=font_size-3)
    plt.xticks(fontsize=font_size-8)
    plt.yticks(fontsize=font_size-8)
    plt.legend(fontsize=font_size-8, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    figure_6_file = os.path.join(figure_dir, "06_trajectories_step10.png")
    plt.savefig(figure_6_file, dpi=300, bbox_inches='tight')
    print(f"第六张图已保存至: {figure_6_file}")
    plt.show()

# 15. 输出改进的韧性动力学方程
print("\n=== 改进的韧性动力学方程（进口和出口分离，使用B_i，分母改为x_mean，随机去边）===")
print("原始方程:")
print("dx_i/dt = B_i * x_i + Σ_j E[i,j] * (x_in[j]**2 / (x_in[j]**2 + C1)) + Σ_j G[i,j] * (x_out[j]**2 / (x_out[j]**2 + C2))")
print(f"\n降维后方程:")
print(f"B * x_mean + alpha * (x_in_mean**2 / (x_in_mean**2 + C1)) + beta * (x_out_mean**2 / (x_out_mean**2 + C2)) = 0")
print(f"其中 alpha = {x_in_mean:.6f}, beta = {x_out_mean:.6f}, x_mean = {x_mean:.6f}")
print(f"求解得到的参数 B (降维方程) = {B_solution:.6f}")
print(f"B_i的平均值 = {B_mean:.6f}")
print(f"B_i的范围: [{np.min(B_i_values):.6f}, {np.max(B_i_values):.6f}]")

# 16. 保存网络基本信息
network_info = {
    'Original_Number_of_Nodes': G.number_of_nodes(),
    'Original_Number_of_Edges': G.number_of_edges(),
    'LCC_Number_of_Nodes': G_lcc.number_of_nodes(),
    'LCC_Number_of_Edges': G_lcc.number_of_edges(),
    'Min_Trade_Volume': min(volumes),
    'Max_Trade_Volume': max(volumes),
    'Total_Weight': np.sum(volumes),
    'Mean_Trade_Volume': x_mean,
    'Mean_Import_Volume': x_in_mean,
    'Mean_Export_Volume': x_out_mean,
    'Parameter_B': B_solution, 
    'Parameter_B_mean': B_mean,
    'Parameter_B_min': np.min(B_i_values),
    'Parameter_B_max': np.max(B_i_values),
    'Total_Edge_Weight': total_edge_weight,
    'Number_of_Edges': len(edges),
}

info_df = pd.DataFrame([network_info])
info_file = os.path.join(output_dir, "network_information_edge_removal.csv")
info_df.to_csv(info_file, index=False)
print(f"网络信息已保存至: {info_file}")

print("\n分析完成！") 