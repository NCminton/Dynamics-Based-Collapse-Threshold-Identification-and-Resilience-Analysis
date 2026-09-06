# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:25:48 2026

@author: mengx
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 22:13:19 2026

@author: mengx
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 15:52:51 2026

@author: mengxueyu

修改说明：
1. 分析2006-2024年的航空发动机供应链网络韧性
2. 将所有文本改为英文，避免中文字体警告
3. 将随机去边方式改为按照连边权重从大到小的顺序去边
4. 修改输出目录为指定的路径
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

# ================ 修改部分：设置年份范围和路径 ================
# 设置要分析的年份范围（从1994年到2024年）
years = list(range(1994, 2025))  # 1994-2024年

# ================ 修改部分：修改输出目录为权重从大到小去边的路径 ================
# 设置路径（已修改为指定的权重从大到小去边路径）
data_dir = r"E:\论文写作\35 TRE\2 每年原始数据\3 网络数据_新"
output_dir = r"E:\论文写作\35 TRE\代码\5 30年分析结果_去边\新的数据集的崩溃点"
figure_dir = r"E:\论文写作\35 TRE\代码\5 30年分析结果_去边\新的数据集的崩溃点"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

# 存储所有年份的崩溃点结果
collapse_points_summary = []

# ================ 修改部分：循环分析多个年份 ================
for year in years:
    print(f"\n{'='*80}")
    print(f"Starting analysis for {year} aircraft engine supply chain network")
    print(f"{'='*80}")
    
    # 为每个年份创建子目录
    year_output_dir = os.path.join(output_dir, str(year))
    year_figure_dir = os.path.join(figure_dir, str(year))
    os.makedirs(year_output_dir, exist_ok=True)
    os.makedirs(year_figure_dir, exist_ok=True)
    
    # 1. 读取指定年份的网络数据并立即对权重进行对数变换
    def load_network(year):
        """Load network data for specified year, construct directed weighted graph, and apply logarithmic transformation"""
        file_path = os.path.join(data_dir, f"edgelist_{year}.csv")
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist, skipping {year} analysis")
            return None
        
        df = pd.read_csv(file_path)
        
        G = nx.DiGraph()
        for _, row in df.iterrows():
            # Apply logarithmic transformation log10(x+1)
            log_weight = np.log(row['weight'] + 1)
            G.add_edge(row['source'], row['target'], weight=log_weight)
        
        return G
    
    G = load_network(year)
    if G is None:
        continue
    
    print(f"{year} original network nodes: {G.number_of_nodes()}")
    print(f"{year} original network edges: {G.number_of_edges()}")
    
    # Extract largest strongly connected component
    strongly_connected_components = list(nx.strongly_connected_components(G))
    largest_scc = max(strongly_connected_components, key=len)
    G_lcc = G.subgraph(largest_scc).copy()
    G_lcc.remove_edges_from(nx.selfloop_edges(G_lcc))
    
    print(f"{year} largest strongly connected component nodes: {G_lcc.number_of_nodes()}")
    print(f"{year} largest strongly connected component edges: {G_lcc.number_of_edges()}")
    
    # Calculate import and export volumes for each country - using log-transformed weights
    node_in_volumes = {}
    node_out_volumes = {}
    node_trade_volumes = {}
    
    for node in G_lcc.nodes():
        in_volume = sum([G_lcc[u][node]['weight'] for u in G_lcc.predecessors(node)])
        out_volume = sum([G_lcc[node][v]['weight'] for v in G_lcc.successors(node)])
        node_in_volumes[node] = in_volume
        node_out_volumes[node] = out_volume
        node_trade_volumes[node] = in_volume + out_volume
    
    # Output log-transformed trade volume ranges
    in_volumes = list(node_in_volumes.values())
    out_volumes = list(node_out_volumes.values())
    volumes = list(node_trade_volumes.values())
    
    print(f"{year} import volume range: [{min(in_volumes):.2f}, {max(in_volumes):.2f}]")
    print(f"{year} export volume range: [{min(out_volumes):.2f}, {max(out_volumes):.2f}]")
    print(f"{year} total trade volume range: [{min(volumes):.2f}, {max(volumes):.2f}]")
    
    # Build import matrix E and export matrix G
    nodes = list(G_lcc.nodes())
    n = len(nodes)
    
    # Create node to index mapping
    node_to_index = {node: i for i, node in enumerate(nodes)}
    
    # Import matrix E: E[i,j] represents import weight from j to i
    # Export matrix G: G[i,j] represents export weight from i to j
    E = np.zeros((n, n))  # Import matrix
    G_matrix = np.zeros((n, n))  # Export matrix (renamed to avoid conflict with network G)
    
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i != j and G_lcc.has_edge(node_j, node_i):  # Note direction: from j to i is import
                E[i, j] = G_lcc[node_j][node_i]['weight']
            if i != j and G_lcc.has_edge(node_i, node_j):  # From i to j is export
                G_matrix[i, j] = G_lcc[node_i][node_j]['weight']
    
    print(f"{year} import matrix E non-zero elements: {np.count_nonzero(E)}")
    print(f"{year} export matrix G non-zero elements: {np.count_nonzero(G_matrix)}")
    
    # Calculate average parameters
    x_in_values = np.array([node_in_volumes[node] for node in nodes])
    x_out_values = np.array([node_out_volumes[node] for node in nodes])
    x_values = np.array([node_trade_volumes[node] for node in nodes])
    
    x_in_mean = np.mean(x_in_values)  # Average import volume
    x_out_mean = np.mean(x_out_values)  # Average export volume
    x_mean = np.mean(x_values)   # Average trade volume
    C1 = x_in_mean
    C2 = x_out_mean
    
    print(f"{year} average import volume <x_in>: {x_in_mean}")
    print(f"{year} average export volume <x_out>: {x_out_mean}")
    print(f"{year} average trade volume <x>: {x_mean}")
    
    # 3. Solve parameter B - using improved dimension-reduced equation
    def equation_for_B(B):
        """Improved dimension-reduced resilience dynamics equation - denominator changed to x_mean"""
        alpha = x_in_mean
        beta = x_out_mean
        return B * x_mean + alpha * (x_in_mean**2 / (x_in_mean**2 + C1)) + beta * (x_out_mean**2 / (x_out_mean**2 + C2))
    
    # Solve for B using numerical methods
    B_solution = fsolve(equation_for_B, -0.1)[0]
    print(f"{year} solved parameter B (dimension-reduced equation): {B_solution}")
    
    # 4. Calculate B_i for each node
    def compute_B_i_for_each_node(nodes, node_in_volumes, node_out_volumes, E, G_matrix, x_mean):
        """Calculate B_i value for each node - denominator changed to x_mean"""
        n = len(nodes)
        B_i_values = np.zeros(n)
        
        for i, node in enumerate(nodes):  
            # Get import and export volumes for node i
            x_in_i = node_in_volumes[node]
            x_out_i = node_out_volumes[node]
            x_i = x_in_i + x_out_i
            
            if x_i == 0:
                B_i_values[i] = 0
                continue
                
            # Calculate import interaction term and export interaction term for node i - denominator changed to x_mean
            import_interaction = 0
            export_interaction = 0
            
            for j in range(n):
                if i != j:
                    # Import interaction: import from j to i
                    if E[i, j] > 0:
                        x_in_j = node_in_volumes[nodes[j]]
                        import_interaction += E[i, j] * (x_in_j**2 / (x_in_j**2 + C1))
                    
                    # Export interaction: export from i to j
                    if G_matrix[i, j] > 0:
                        x_out_j = node_out_volumes[nodes[j]]
                        export_interaction += G_matrix[i, j] * (x_out_j**2 / (x_out_j**2 + C2))
            
            # Solve for B_i: B_i * x_i + import_interaction + export_interaction = 0
            if x_i != 0:
                B_i_values[i] = -(import_interaction + export_interaction) / x_i
            else:
                B_i_values[i] = 0
        
        return B_i_values
    
    # Calculate B_i for each node - denominator changed to x_mean
    B_i_values = compute_B_i_for_each_node(nodes, node_in_volumes, node_out_volumes, E, G_matrix, x_mean)
    B_mean = np.mean(B_i_values)
    print(f"{year} average B_i: {B_mean}")
    print(f"{year} B_i range: [{np.min(B_i_values):.6f}, {np.max(B_i_values):.6f}]")
    
    # Save B_i values
    B_i_df = pd.DataFrame({
        'Node': nodes,
        'B_i': B_i_values,
        'In_Volume': [node_in_volumes[node] for node in nodes],
        'Out_Volume': [node_out_volumes[node] for node in nodes],
        'Total_Volume': [node_trade_volumes[node] for node in nodes]
    })
    B_i_file = os.path.join(year_output_dir, "B_i_values.csv")
    B_i_df.to_csv(B_i_file, index=False)
    print(f"{year} B_i values saved to: {B_i_file}")
    
    # ================ 修改部分1：改为按照连边权重从大到小的顺序去边 ================
    # Get all edges and their weights
    edges = list(G_lcc.edges(data=True))
    edge_weights = [edge[2]['weight'] for edge in edges]
    total_edge_weight = sum(edge_weights)
    
    print(f"{year} total edge weight: {total_edge_weight}")
    print(f"{year} number of edges: {len(edges)}")
    
    # ================ 修改：按照连边权重从大到小的顺序排序 ================
    # Sort edges by weight in descending order
    edges_sorted_by_weight = sorted(edges, key=lambda x: x[2]['weight'], reverse=True)
    
    # 保存权重从大到小的去边顺序
    edge_removal_order_df = pd.DataFrame({
        'Removal_Order': range(1, len(edges_sorted_by_weight) + 1),
        'Source': [edge[0] for edge in edges_sorted_by_weight],
        'Target': [edge[1] for edge in edges_sorted_by_weight],
        'Weight': [edge[2]['weight'] for edge in edges_sorted_by_weight]
    })
    removal_order_file = os.path.join(year_output_dir, "weight_descending_edge_removal_order.csv")
    edge_removal_order_df.to_csv(removal_order_file, index=False)
    print(f"{year} weight-descending edge removal order saved to: {removal_order_file}")
    
    # 5. Improved resilience analysis - theoretical analytical method (using average B) - modified for edge removal
    def theoretical_robustness_analysis_edge_removal(G_lcc, nodes, node_in_volumes, node_out_volumes, B, x_mean, removal_order_edges):
        """Improved theoretical analytical resilience analysis - using separated import and export dynamics, denominator changed to x_mean, edge removal by weight descending"""
        theoretical_results_in = []  # Steady-state solutions for import part
        theoretical_results_out = []  # Steady-state solutions for export part
        theoretical_results = []  # Steady-state solutions for total trade volume
        steady_state_solutions_in = []  # Steady-state solutions for import part
        steady_state_solutions_out = []  # Steady-state solutions for export part
        steady_state_solutions = []  # Steady-state solutions for total trade volume
        alpha_values = []  # Store alpha for each q value
        beta_values = []  # Store beta for each q value
        q_values = []  # Store q value for each removal step
        
        # Initial network
        G_current = G_lcc.copy()
        
        # Calculate initial weight proportion
        removed_weight = 0
        
        # Initial state: no edges removed
        q = 0.0
        q_values.append(q)
        
        # Calculate average import and export volumes of current remaining network
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
        
        # Solve steady-state equations for import and export separately - denominator changed to x_mean
        def steady_state_eq_in(x_in):
            return B * x_in + alpha_current * (x_in**2 / (x_in**2 + C1))
        
        def steady_state_eq_out(x_out):
            return B * x_out + beta_current * (x_out**2 / (x_out**2 + C2))
        
        # Find non-zero solution for import part
        if alpha_current > 0:
            # Try multiple initial values to find stable solution
            solutions_in = []
            for initial_guess in [0.1, x_in_mean, 2*x_in_mean, 5*x_in_mean]:
                try:
                    x_solution = fsolve(steady_state_eq_in, initial_guess)[0]
                    if x_solution > 0:  # Only consider positive solutions
                        solutions_in.append(x_solution)
                except:
                    continue
            
            if solutions_in:
                # Select the largest positive solution (corresponding to stable state)
                x_solution_in = max(solutions_in)
                steady_state_solutions_in.append(x_solution_in)
                theoretical_results_in.append(x_solution_in)
            else:
                steady_state_solutions_in.append(0)
                theoretical_results_in.append(0)
        else:
            steady_state_solutions_in.append(0)
            theoretical_results_in.append(0)
        
        # Find non-zero solution for export part
        if beta_current > 0:
            # Try multiple initial values to find stable solution
            solutions_out = []
            for initial_guess in [0.1, x_out_mean, 2*x_out_mean, 5*x_out_mean]:
                try:
                    x_solution = fsolve(steady_state_eq_out, initial_guess)[0]
                    if x_solution > 0:  # Only consider positive solutions
                        solutions_out.append(x_solution)
                except:
                    continue
            
            if solutions_out:
                # Select the largest positive solution (corresponding to stable state)
                x_solution_out = max(solutions_out)
                steady_state_solutions_out.append(x_solution_out)
                theoretical_results_out.append(x_solution_out)
            else:
                steady_state_solutions_out.append(0)
                theoretical_results_out.append(0)
        else:
            steady_state_solutions_out.append(0)
            theoretical_results_out.append(0)
        
        # Total trade volume is sum of import and export
        total_solution = steady_state_solutions_in[-1] + steady_state_solutions_out[-1]
        steady_state_solutions.append(total_solution)
        theoretical_results.append(total_solution)
        
        print(f"{year} initial state - alpha: {alpha_current:.4f}, beta: {beta_current:.4f}")
        print(f"{year} import solution: {steady_state_solutions_in[-1]:.4f}, export solution: {steady_state_solutions_out[-1]:.4f}, total solution: {total_solution:.4f}")
        
        # Remove edges according to weight-descending order
        for step, (u, v, attr) in enumerate(removal_order_edges):
            if not G_current.has_edge(u, v):
                continue
                
            # Remove this edge from current network
            edge_weight = G_current[u][v]['weight']
            G_current.remove_edge(u, v)
            
            # Update removed weight
            removed_weight += edge_weight
            
            # Calculate current removal rate q (based on edge weight)
            q = removed_weight / total_edge_weight
            q_values.append(q)
            
            # Calculate average import and export volumes of current remaining network
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
            
            # Solve steady-state equations for import and export separately - denominator changed to x_mean
            def steady_state_eq_in(x_in):
                return B * x_in + alpha_current * (x_in**2 / (x_in**2 + C1))
            
            def steady_state_eq_out(x_out):
                return B * x_out + beta_current * (x_out**2 / (x_out**2 + C2))
            
            # Find non-zero solution for import part
            if alpha_current > 0:
                # Try multiple initial values to find stable solution
                solutions_in = []
                for initial_guess in np.linspace(0, 1000, 1000):
                    try:
                        x_solution = fsolve(steady_state_eq_in, initial_guess)[0]
                        if x_solution > 0:  # Only consider positive solutions
                            solutions_in.append(x_solution)
                    except:
                        continue
                
                if solutions_in:
                    # Select the largest positive solution (corresponding to stable state)
                    x_solution_in = max(solutions_in)
                    steady_state_solutions_in.append(x_solution_in)
                    theoretical_results_in.append(x_solution_in) 
                else:
                    steady_state_solutions_in.append(0)
                    theoretical_results_in.append(0)
            else:
                steady_state_solutions_in.append(0)
                theoretical_results_in.append(0)
            
            # Find non-zero solution for export part
            if beta_current > 0:
                # Try multiple initial values to find stable solution
                solutions_out = []
                for initial_guess in np.linspace(0, 1000, 1000):
                    try:
                        x_solution = fsolve(steady_state_eq_out, initial_guess)[0]
                        if x_solution > 0:  # Only consider positive solutions
                            solutions_out.append(x_solution)
                    except:
                        continue
                
                if solutions_out:
                    # Select the largest positive solution (corresponding to stable state)
                    x_solution_out = max(solutions_out)
                    steady_state_solutions_out.append(x_solution_out)
                    theoretical_results_out.append(x_solution_out)
                else:
                    steady_state_solutions_out.append(0)
                    theoretical_results_out.append(0)
            else:
                steady_state_solutions_out.append(0)
                theoretical_results_out.append(0)
            
            # Total trade volume is sum of import and export
            total_solution = steady_state_solutions_in[-1] + steady_state_solutions_out[-1]
            steady_state_solutions.append(total_solution)
            theoretical_results.append(total_solution)
            
            if (step + 1) % 30 == 0:
                print(f"{year} step {step+1}/{len(removal_order_edges)} - removed edge: ({u}, {v}), weight: {edge_weight:.4f}")
                print(f"{year} alpha: {alpha_current:.4f}, beta: {beta_current:.4f}")
                print(f"{year} import solution: {steady_state_solutions_in[-1]:.4f}, export solution: {steady_state_solutions_out[-1]:.4f}, total solution: {total_solution:.4f}")
                print(f"{year} removal ratio q: {q:.4f}")
        
        return q_values, theoretical_results, steady_state_solutions, steady_state_solutions_in, steady_state_solutions_out, alpha_values, beta_values
    
    # 6. Define improved dynamics equation functions - separate import and export, using B_i, denominator changed to x_mean
    def dynamics_import(t, x_in, B_vector, E_sub, node_indices, x_mean):
        """Import dynamics equation function for solve_ivp, using B_i, denominator changed to x_mean"""
        dxdt_in = np.zeros_like(x_in)
        n_nodes = len(x_in)
        
        for i in range(n_nodes):
            # Get B_i for current node
            original_idx = node_indices[i]
            B_i = B_vector[original_idx]
            
            # Self dynamics
            dxdt_in[i] = B_i * x_in[i]
            
            # Import interaction term - denominator changed to x_mean
            import_interaction = 0
            for j in range(n_nodes):
                if i != j and E_sub[i, j] > 0:
                    import_interaction += E_sub[i, j] * (x_in[j]**2 / (x_in[j]**2 + C1))
            
            dxdt_in[i] += import_interaction
        
        return dxdt_in
    
    def dynamics_export(t, x_out, B_vector, G_sub, node_indices, x_mean):
        """Export dynamics equation function for solve_ivp, using B_i, denominator changed to x_mean"""
        dxdt_out = np.zeros_like(x_out)
        n_nodes = len(x_out)
        
        for i in range(n_nodes):
            # Get B_i for current node
            original_idx = node_indices[i]
            B_i = B_vector[original_idx]
            
            # Self dynamics
            dxdt_out[i] = B_i * x_out[i]
            
            # Export interaction term - denominator changed to x_mean
            export_interaction = 0
            for j in range(n_nodes):
                if i != j and G_sub[i, j] > 0:
                    export_interaction += G_sub[i, j] * (x_out[j]**2 / (x_out[j]**2 + C2))
            
            dxdt_out[i] += export_interaction
        
        return dxdt_out
    
    # 7. Improved resilience analysis - simulation method based on dynamics equations (separate import and export, using B_i, denominator changed to x_mean, edge removal by weight descending)
    def simulate_dynamics_robustness_edge_removal(G_lcc, nodes, node_in_volumes, node_out_volumes, E, G_matrix, B_i_values, x_mean, removal_order_edges):
        """Simulation resilience analysis based on improved dynamics equations - separate import and export, using B_i, denominator changed to x_mean, edge removal by weight descending"""
        n = len(nodes)
        
        # Create node to index mapping
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Initial matrices
        E_current = E.copy()
        G_current = G_matrix.copy()
        
        # Record current edge state (which edges are still present)
        current_edges = set(G_lcc.edges())
        
        # Record evolution curves for all countries at each q value (import)
        all_country_steady_states_in = []
        all_country_trajectories_in = []
        
        # Record evolution curves for all countries at each q value (export)
        all_country_steady_states_out = []
        all_country_trajectories_out = []
        
        # Record total trade volume evolution curves for all countries at each q value
        all_country_steady_states = []
        all_country_trajectories = []
        
        # Record network average import and export volumes at each q value (calculated from network structure)
        simulation_alpha_values = []  # Network average import volume alpha (theoretical parameter)
        simulation_beta_values = []   # Network average export volume beta (theoretical parameter)
        
        # Record simulated steady-state averages at each q value
        simulation_avg_in_steady = []  # Simulated import steady-state average
        simulation_avg_out_steady = []  # Simulated export steady-state average
        simulation_avg_total_steady = []  # Simulated total trade volume steady-state average
        
        # Numerical integration parameters
        t_span = (0, 100)  # Time range
        t_eval = np.linspace(0, 100, 100)  # Evaluation time points
        steady_window = 10  # Number of final time steps for calculating steady-state value
        
        # Calculate initial weight proportion
        removed_weight = 0
        q_values = []
        
        # Use progress bar
        print(f"{year} starting edge removal simulation (weight descending order)...")
        pbar = tqdm(total=len(removal_order_edges) + 1, desc=f"{year} edge removal progress")
        
        # Initial state: no edges removed
        q = 0.0
        q_values.append(q)
        
        # Calculate current network average import and export volumes (network structure parameters)
        # Create temporary graph to calculate current weighted degree
        temp_G = nx.DiGraph()
        for node in nodes:
            temp_G.add_node(node)
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if E_current[i, j] > 0:
                    temp_G.add_edge(node_j, node_i, weight=E_current[i, j])
                if G_current[i, j] > 0:
                    temp_G.add_edge(node_i, node_j, weight=G_current[i, j])
        
        # Calculate current weighted degree
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
        
        # Initial state (log-transformed import and export volumes)
        x0_in = np.array(current_in_volumes)  # Use current network weighted degree as initial value
        x0_out = np.array(current_out_volumes)  # Use current network weighted degree as initial value
        
        # Initial simulation: all edges present
        try:
            # Solve import dynamics - denominator changed to x_mean
            sol_in = solve_ivp(
                dynamics_import, 
                t_span, 
                x0_in, 
                args=(B_i_values, E_current, list(range(n)), x_mean),
                t_eval=t_eval,
                method='RK45'
            )
            
            # Solve export dynamics - denominator changed to x_mean
            sol_out = solve_ivp(
                dynamics_export, 
                t_span, 
                x0_out, 
                args=(B_i_values, G_current, list(range(n)), x_mean),
                t_eval=t_eval,
                method='RK45'
            )
            
            if sol_in.success and sol_out.success:
                x_trajectory_in = sol_in.y.T  # Transpose to (time, nodes) shape
                x_trajectory_out = sol_out.y.T
                
                # Calculate total trade volume evolution curve
                x_trajectory_total = x_trajectory_in + x_trajectory_out
                
                # Take average of last few steps as steady-state value
                steady_states_in = np.mean(x_trajectory_in[-steady_window:], axis=0)
                steady_states_out = np.mean(x_trajectory_out[-steady_window:], axis=0)
                steady_states_total = steady_states_in + steady_states_out
                
                # Calculate steady-state averages
                avg_in_steady = np.mean(steady_states_in)
                avg_out_steady = np.mean(steady_states_out)
                avg_total_steady = np.mean(steady_states_total)
                
                # Save steady-state values for all countries
                all_country_steady_states_in.append(steady_states_in)
                all_country_steady_states_out.append(steady_states_out)
                all_country_steady_states.append(steady_states_total)
                
                # Save evolution curves
                country_trajectories = x_trajectory_total
                all_country_trajectories.append(country_trajectories)
                
                # Save simulated steady-state averages
                simulation_avg_in_steady.append(avg_in_steady)
                simulation_avg_out_steady.append(avg_out_steady)
                simulation_avg_total_steady.append(avg_total_steady)
                
            else:
                print(f'{year} initial state solution failed')
                # Use initial values as steady-state values
                all_country_steady_states_in.append(x0_in)
                all_country_steady_states_out.append(x0_out)
                all_country_steady_states.append(x0_in + x0_out)
                country_trajectories = np.zeros((len(t_eval), n)) * np.nan
                all_country_trajectories.append(country_trajectories)
                
                # Save simulated steady-state averages (using initial values)
                simulation_avg_in_steady.append(np.mean(x0_in))
                simulation_avg_out_steady.append(np.mean(x0_out))
                simulation_avg_total_steady.append(np.mean(x0_in + x0_out))
        except:
            print(f'{year} initial state solution failed')
            # Use initial values as steady-state values
            all_country_steady_states_in.append(x0_in)
            all_country_steady_states_out.append(x0_out)
            all_country_steady_states.append(x0_in + x0_out)
            country_trajectories = np.zeros((len(t_eval), n)) * np.nan
            all_country_trajectories.append(country_trajectories)
            
            # Save simulated steady-state averages (using initial values)
            simulation_avg_in_steady.append(np.mean(x0_in))
            simulation_avg_out_steady.append(np.mean(x0_out))
            simulation_avg_total_steady.append(np.mean(x0_in + x0_out))
        
        # Update progress bar
        pbar.update(1)
        
        # Remove edges according to weight-descending order
        for step, (u, v, attr) in enumerate(removal_order_edges):
            if (u, v) not in current_edges:
                # Edge already removed, skip
                # But still need to record q value
                edge_weight = attr['weight']
                removed_weight += edge_weight
                q = removed_weight / total_edge_weight
                q_values.append(q)
                
                # Repeat steady-state values from previous state
                all_country_steady_states_in.append(all_country_steady_states_in[-1])
                all_country_steady_states_out.append(all_country_steady_states_out[-1])
                all_country_steady_states.append(all_country_steady_states[-1])
                all_country_trajectories.append(all_country_trajectories[-1])
                
                # Repeat network structure parameters and simulated steady-state averages
                simulation_alpha_values.append(simulation_alpha_values[-1])
                simulation_beta_values.append(simulation_beta_values[-1])
                simulation_avg_in_steady.append(simulation_avg_in_steady[-1])
                simulation_avg_out_steady.append(simulation_avg_out_steady[-1])
                simulation_avg_total_steady.append(simulation_avg_total_steady[-1])
                
                pbar.update(1)
                continue
            
            # Get edge weight
            edge_weight = attr['weight']
            
            # Remove this edge from current edge set
            current_edges.remove((u, v))
            
            # Update matrices
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            
            # If this is an import edge (from v to u)
            if E_current[v_idx, u_idx] > 0:
                E_current[v_idx, u_idx] = 0
            
            # If this is an export edge (from u to v)
            if G_current[u_idx, v_idx] > 0:
                G_current[u_idx, v_idx] = 0
            
            # Update removed weight
            removed_weight += edge_weight
            
            # Calculate current removal rate q (based on edge weight)
            q = removed_weight / total_edge_weight
            q_values.append(q)
            
            # Update dynamic initial values: recalculate current network weighted degree
            # Create temporary graph to calculate current weighted degree
            temp_G = nx.DiGraph()
            
            # First add all nodes to ensure they exist
            for node in nodes:
                temp_G.add_node(node)
            
            # Then add edges
            for i, node_i in enumerate(nodes):
                for j, node_j in enumerate(nodes):
                    if E_current[i, j] > 0:
                        temp_G.add_edge(node_j, node_i, weight=E_current[i, j])
                    if G_current[i, j] > 0:
                        temp_G.add_edge(node_i, node_j, weight=G_current[i, j])
            
            # Calculate current weighted degree - fix node access issue
            current_in_volumes = []
            current_out_volumes = []
            for node in nodes:
                # Check if node is in graph (should be, since we added all nodes)
                if node in temp_G.nodes():
                    # Calculate in-degree (import volume)
                    in_vol = sum([temp_G[u][node]['weight'] for u in temp_G.predecessors(node)])
                    # Calculate out-degree (export volume)
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
            
            # Update initial values
            x0_in = np.array(current_in_volumes)
            x0_out = np.array(current_out_volumes)
            
            # Perform simulation
            try:
                # Solve import dynamics - denominator changed to x_mean
                sol_in = solve_ivp(
                    dynamics_import, 
                    t_span, 
                    x0_in, 
                    args=(B_i_values, E_current, list(range(n)), x_mean),
                    t_eval=t_eval,
                    method='RK45'
                )
                
                # Solve export dynamics - denominator changed to x_mean
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
                    
                    # Take average of last few steps as steady-state value
                    steady_states_in = np.mean(x_trajectory_in[-steady_window:], axis=0)
                    steady_states_out = np.mean(x_trajectory_out[-steady_window:], axis=0)
                    steady_states_total = steady_states_in + steady_states_out
                    
                    # Calculate steady-state averages
                    avg_in_steady = np.mean(steady_states_in)
                    avg_out_steady = np.mean(steady_states_out)
                    avg_total_steady = np.mean(steady_states_total)
                    
                    # Save steady-state values for all countries
                    all_country_steady_states_in.append(steady_states_in)
                    all_country_steady_states_out.append(steady_states_out)
                    all_country_steady_states.append(steady_states_total)
                    
                    # Save evolution curves
                    country_trajectories = x_trajectory_total
                    all_country_trajectories.append(country_trajectories)
                    
                    # Save simulated steady-state averages
                    simulation_avg_in_steady.append(avg_in_steady)
                    simulation_avg_out_steady.append(avg_out_steady)
                    simulation_avg_total_steady.append(avg_total_steady)
                    
                else:
                    print(f'{year} step {step+1} solution failed')
                    # Use current initial values as steady-state values
                    all_country_steady_states_in.append(x0_in)
                    all_country_steady_states_out.append(x0_out)
                    all_country_steady_states.append(x0_in + x0_out)
                    country_trajectories = np.zeros((len(t_eval), n)) * np.nan
                    all_country_trajectories.append(country_trajectories)
                    
                    # Save simulated steady-state averages (using current initial values)
                    simulation_avg_in_steady.append(np.mean(x0_in))
                    simulation_avg_out_steady.append(np.mean(x0_out))
                    simulation_avg_total_steady.append(np.mean(x0_in + x0_out))
            except:
                print(f'{year} step {step+1} solution failed')
                # Use current initial values as steady-state values
                all_country_steady_states_in.append(x0_in)
                all_country_steady_states_out.append(x0_out)
                all_country_steady_states.append(x0_in + x0_out)
                country_trajectories = np.zeros((len(t_eval), n)) * np.nan
                all_country_trajectories.append(country_trajectories)
                
                # Save simulated steady-state averages (using current initial values)
                simulation_avg_in_steady.append(np.mean(x0_in))
                simulation_avg_out_steady.append(np.mean(x0_out))
                simulation_avg_total_steady.append(np.mean(x0_in + x0_out))
            
            # Update progress bar
            pbar.update(1)
        
        pbar.close()
        
        return (q_values, simulation_avg_total_steady, simulation_avg_in_steady, simulation_avg_out_steady,
                all_country_steady_states, all_country_steady_states_in, all_country_steady_states_out,
                all_country_trajectories, simulation_alpha_values, simulation_beta_values)
    
    # 8. Execute resilience analysis
    print(f"{year} starting improved theoretical analytical resilience analysis (weight descending edge removal)...")
    theoretical_q_values, theoretical_results_total, steady_state_solutions_total, steady_state_solutions_in, steady_state_solutions_out, alpha_values, beta_values = theoretical_robustness_analysis_edge_removal(
        G_lcc, nodes, node_in_volumes, node_out_volumes, B_solution, x_mean, edges_sorted_by_weight)
    
    print(f"{year} starting simulation resilience analysis based on improved dynamics equations (weight descending edge removal)...")
    simulation_q_values, simulation_avg_total_steady, simulation_avg_in_steady, simulation_avg_out_steady, all_country_steady_states_total, all_country_steady_states_in, all_country_steady_states_out, all_country_trajectories_total, simulation_alpha_values, simulation_beta_values = simulate_dynamics_robustness_edge_removal(
        G_lcc, nodes, node_in_volumes, node_out_volumes, E, G_matrix, B_i_values, x_mean, edges_sorted_by_weight)
    
    # 9. Save results
    # Save resilience analysis results
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
    
    results_file = os.path.join(year_output_dir, "resilience_analysis_results_edge_removal.csv")
    results_df.to_csv(results_file, index=False)
    print(f"{year} resilience analysis results saved to: {results_file}")
    
    # Save solutions for each q value
    solutions_df = pd.DataFrame({
        'Removal_Ratio': theoretical_q_values,
        'Steady_State_Solution_Total': theoretical_results_total,
        'Steady_State_Solution_In': steady_state_solutions_in,
        'Steady_State_Solution_Out': steady_state_solutions_out,
        'Alpha': alpha_values,
        'Beta': beta_values
    })
    
    solutions_file = os.path.join(year_output_dir, "steady_state_solutions_edge_removal.csv")
    solutions_df.to_csv(solutions_file, index=False)
    print(f"{year} steady-state solutions saved to: {solutions_file}")
    
    # Save country curves vs q
    country_curves_total_df = pd.DataFrame(all_country_steady_states_total, index=simulation_q_values)
    country_curves_total_df.columns = nodes
    country_curves_file = os.path.join(year_output_dir, "country_curves_vs_q_edge_removal.csv")
    country_curves_total_df.to_csv(country_curves_file)
    print(f"{year} country curves vs q saved to: {country_curves_file}")
    
    # ================ 修改部分：检测并记录崩溃点 ================
    def detect_collapse_point(q_values, theoretical_results, threshold_tol=1e-3):
        """Detect collapse point (critical point where steady-state value changes from >0 to 0)"""
        collapse_q = None
        collapse_index = None
        
        if len(q_values) > 1 and len(theoretical_results) > 1:
            # Find critical point where steady-state value changes from >0 to 0 or from 0 to >0
            for i in range(1, len(theoretical_results)):
                if theoretical_results[i-1] > threshold_tol and abs(theoretical_results[i]) < threshold_tol:
                    collapse_q = q_values[i]
                    collapse_index = i
                    break
                elif abs(theoretical_results[i-1]) < threshold_tol and theoretical_results[i] > threshold_tol:
                    collapse_q = q_values[i]
                    collapse_index = i
                    break
        
        return collapse_q, collapse_index
    
    # Detect collapse points for total steady-state, import steady-state, and export steady-state
    collapse_q_total, collapse_idx_total = detect_collapse_point(theoretical_q_values, theoretical_results_total)
    collapse_q_in, collapse_idx_in = detect_collapse_point(theoretical_q_values, steady_state_solutions_in)
    collapse_q_out, collapse_idx_out = detect_collapse_point(theoretical_q_values, steady_state_solutions_out)
    
    # Record collapse points
    collapse_points_summary.append({
        'Year': year,
        'Nodes': G_lcc.number_of_nodes(),
        'Edges': G_lcc.number_of_edges(),
        'Total_Edge_Weight': total_edge_weight,
        'Mean_Trade_Volume': x_mean,
        'Parameter_B': B_solution,
        'Collapse_q_total': collapse_q_total,
        'Collapse_q_in': collapse_q_in,
        'Collapse_q_out': collapse_q_out,
        'Collapse_idx_total': collapse_idx_total,
        'Collapse_idx_in': collapse_idx_in,
        'Collapse_idx_out': collapse_idx_out,
        'Max_Steady_Total': np.max(theoretical_results_total) if len(theoretical_results_total) > 0 else 0,
        'Max_Steady_In': np.max(steady_state_solutions_in) if len(steady_state_solutions_in) > 0 else 0,
        'Max_Steady_Out': np.max(steady_state_solutions_out) if len(steady_state_solutions_out) > 0 else 0
    })
    
    print(f"{year} collapse point detection results:")
    print(f"  Total steady-state collapse point q_e*: {collapse_q_total}")
    print(f"  Import steady-state collapse point q_e,in*: {collapse_q_in}")
    print(f"  Export steady-state collapse point q_e,out*: {collapse_q_out}")
    
    # ================ 绘图部分（每个年份单独保存） ================
    macaron_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
    
    # 1. Figure 1: Network total steady-state value <x> vs edge weight removal ratio q_e
    print(f"\nPlotting {year} Figure 1: Network total steady-state value <x> vs q_e...")
    
    plt.figure(figsize=(14, 10))
    
    # Plot theoretical curve (total trade volume)
    theoretical_plot_values_total = [max(val, 1e-5) for val in theoretical_results_total]
    plt.plot(theoretical_q_values, theoretical_plot_values_total, 
             label='Theory-based ODE', color=macaron_colors[1], linewidth=8, linestyle='-')
    
    # Plot simulation points (total trade volume)
    plt.scatter(simulation_q_values, [max(val, 1e-5) for val in simulation_avg_total_steady],
                label='Simulation-based HDE', color=macaron_colors[0],
                s=100, marker='^', alpha=0.7, zorder=5)
    
    plt.axhline(y=np.max(theoretical_results_total),
                color='gray',      # Color
                linestyle='--',    # Dashed line
                linewidth=3)  
    
    # Mark threshold point
    if collapse_q_total is not None:
        collapse_y_total = theoretical_results_total[collapse_idx_total] if collapse_idx_total is not None else 0
        plt.plot(collapse_q_total, max(collapse_y_total, 1e-5), 'ro', markersize=12, markeredgecolor='red', 
                 markeredgewidth=2, zorder=10)
        
        annotation_text = f'$q_e^{{*}}={collapse_q_total:.3f}$'
        
        # Determine annotation position
        if collapse_q_total > 0.5:
            text_x = collapse_q_total - 0.08
        else:
            text_x = collapse_q_total + 0.08
        
        y_pos = max(collapse_y_total, 1e-5) * 0.1
        if y_pos < 0.1:
            y_pos = 0.1
        
        plt.annotate(annotation_text, 
                     xy=(collapse_q_total - 0.01, max(collapse_y_total, 1e-5)),
                     xytext=(text_x - 0.16, y_pos *0.5),
                     fontsize=font_size-8,
                     color='red',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5, connectionstyle="arc3,rad=0"),
                     bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='red', alpha=0.5),
                     zorder=10)
    
    # Set y-axis to symlog scale
    plt.yscale('symlog', linthresh=0.1)
    
    plt.xlabel(r'$q_e$', fontsize=font_size-3)
    plt.ylabel(r'$\langle x \rangle$', fontsize=font_size-3)
    plt.xticks(fontsize=font_size-5)
    plt.yticks(fontsize=font_size-5)
    leg = plt.legend(fontsize=font_size-5, loc='center left', bbox_to_anchor=(0.08, 0.5),
               labelspacing=0.9,   # Line spacing coefficient, default 0.5
               edgecolor='black')
    leg.get_frame().set_alpha(0.7)        # 0=fully transparent, 1=opaque
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(-0.03, 1000)
    plt.title(f'{year} Aircraft Engine Supply Chain Network Resilience Analysis (Weight Descending Removal)', fontsize=font_size-5)
    plt.tight_layout()
    
    # Save figure
    figure_1_file = os.path.join(year_figure_dir, f"{year}_total_steady_vs_qe.png")
    plt.savefig(figure_1_file, dpi=600, bbox_inches='tight')
    print(f"{year} Figure 1 saved to: {figure_1_file}")
    plt.close()  # Close figure to avoid memory accumulation
    
    # 2. Figure 2: Import and export steady-state vs q_e
    print(f"\nPlotting {year} Figure 2: Import and export steady-state vs q_e...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left subplot: Import steady-state <x_in> vs q_e
    theoretical_plot_values_in = [max(val, 1e-5) for val in steady_state_solutions_in]
    ax1.plot(theoretical_q_values, theoretical_plot_values_in, 
             label='Theory-based ODE', color=macaron_colors[1], linewidth=5, linestyle='-')
    
    ax1.scatter(simulation_q_values, [max(val, 1e-5) for val in simulation_avg_in_steady],
                label='Simulation-based HDE', color=macaron_colors[0],
                s=120, marker='^', alpha=0.8, zorder=5)
    
    ax1.axhline(y=np.max(theoretical_plot_values_in),
                color='gray',      # Color
                linestyle='--',    # Dashed line
                linewidth=3)  
    
    # Mark import threshold point
    if collapse_q_in is not None:
        collapse_y_in = steady_state_solutions_in[collapse_idx_in] if collapse_idx_in is not None else 0
        ax1.plot(collapse_q_in, max(collapse_y_in, 1e-5), 'ro', markersize=12, markeredgecolor='red', 
                 markeredgewidth=2, zorder=10)
        
        annotation_text = f'$q_{{e,in}}^{{*}}={collapse_q_in:.3f}$'
        
        if collapse_q_in > 0.5:
            text_x = collapse_q_in - 0.08
        else:
            text_x = collapse_q_in + 0.08
        
        y_pos = max(collapse_y_in, 1e-5) * 0.5
        if y_pos < 0.1:
            y_pos = 0.1
        
        ax1.annotate(annotation_text, 
                     xy=(collapse_q_in - 0.01, max(collapse_y_in, 1e-5)),
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
    
    # Right subplot: Export steady-state <x_out> vs q_e
    theoretical_plot_values_out = [max(val, 1e-5) for val in steady_state_solutions_out]
    ax2.plot(theoretical_q_values, theoretical_plot_values_out, 
             label='Theory-based ODE', color=macaron_colors[1], linewidth=5, linestyle='-')
    
    ax2.scatter(simulation_q_values, [max(val, 1e-5) for val in simulation_avg_out_steady],
                label='Simulation-based HDE', color=macaron_colors[0],
                s=120, marker='^', alpha=0.8, zorder=5)
    
    ax2.axhline(y=np.max(theoretical_plot_values_out),
                color='gray',      # Color
                linestyle='--',    # Dashed line
                linewidth=3)  
    
    # Mark export threshold point
    if collapse_q_out is not None:
        collapse_y_out = steady_state_solutions_out[collapse_idx_out] if collapse_idx_out is not None else 0
        ax2.plot(collapse_q_out, max(collapse_y_out, 1e-5), 'ro', markersize=12, markeredgecolor='red', 
                 markeredgewidth=2, zorder=10)
        
        annotation_text = f'$q_{{e,out}}^{{*}}={collapse_q_out:.3f}$'
        
        if collapse_q_out > 0.5:
            text_x = collapse_q_out - 0.08
        else:
            text_x = collapse_q_out + 0.08
        
        y_pos = max(collapse_y_out, 1e-5) * 0.5
        if y_pos < 0.1:
            y_pos = 0.1
        
        ax2.annotate(annotation_text, 
                     xy=(collapse_q_out - 0.01, max(collapse_y_out, 1e-5)),
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
    
    plt.suptitle(f'{year} Aircraft Engine Supply Chain Network Resilience Analysis - Import and Export (Weight Descending Removal)', fontsize=font_size-5)
    plt.tight_layout()
    
    # Save figure
    figure_2_file = os.path.join(year_figure_dir, f"{year}_import_export_steady_vs_qe.png")
    plt.savefig(figure_2_file, dpi=600, bbox_inches='tight')
    print(f"{year} Figure 2 saved to: {figure_2_file}")
    plt.close()

# ================ 绘制崩溃点变化曲线 ================
print("\nPlotting collapse points change curve...")

if collapse_points_summary:
    # Create collapse points summary DataFrame
    collapse_df = pd.DataFrame(collapse_points_summary)
    
    # Save collapse points summary data
    collapse_summary_file = os.path.join(output_dir, "collapse_points_summary.csv")
    collapse_df.to_csv(collapse_summary_file, index=False)
    print(f"Collapse points summary data saved to: {collapse_summary_file}")
    
    # Plot collapse points change curve
    plt.figure(figsize=(14, 10))
    
    # Sort by year
    collapse_df = collapse_df.sort_values('Year')
    
    # Plot total steady-state collapse point
    if collapse_df['Collapse_q_total'].notna().any():
        plt.plot(collapse_df['Year'], collapse_df['Collapse_q_total'], 
                 label=r'Total steady-state collapse point $q_e^{*}$', color='#FF6B6B', 
                 marker='o', markersize=12, linewidth=4, linestyle='-')
    
    # Plot import steady-state collapse point
    if collapse_df['Collapse_q_in'].notna().any():
        plt.plot(collapse_df['Year'], collapse_df['Collapse_q_in'], 
                 label=r'Import steady-state collapse point $q_{e,in}^{*}$', color='#4ECDC4', 
                 marker='s', markersize=12, linewidth=4, linestyle='--')
    
    # Plot export steady-state collapse point
    if collapse_df['Collapse_q_out'].notna().any():
        plt.plot(collapse_df['Year'], collapse_df['Collapse_q_out'], 
                 label=r'Export steady-state collapse point $q_{e,out}^{*}$', color='#45B7D1', 
                 marker='^', markersize=12, linewidth=4, linestyle='-.')
    
    plt.xlabel('Year', fontsize=font_size-3)
    plt.ylabel('Collapse point $q_e^{*}$', fontsize=font_size-3)
    plt.xticks(collapse_df['Year'], fontsize=font_size-5)
    plt.yticks(fontsize=font_size-5)
    plt.legend(fontsize=font_size-8, loc='best')
    plt.grid(True, alpha=0.3)
    plt.title('Aircraft Engine Supply Chain Network Collapse Points Change Curve (2006-2024, Weight Descending Removal)', fontsize=font_size-5)
    plt.tight_layout()
    
    # Save collapse points change curve
    collapse_curve_file = os.path.join(figure_dir, "collapse_points_curve.png")
    plt.savefig(collapse_curve_file, dpi=600, bbox_inches='tight')
    print(f"Collapse points change curve saved to: {collapse_curve_file}")
    plt.show()
    
    # Print summary information
    print("\n" + "="*80)
    print("Collapse Points Summary:")
    print("="*80)
    for idx, row in collapse_df.iterrows():
        print(f"{row['Year']}:")
        print(f"  Network scale: {row['Nodes']} nodes, {row['Edges']} edges")
        print(f"  Total edge weight: {row['Total_Edge_Weight']:.2f}")
        print(f"  Average trade volume: {row['Mean_Trade_Volume']:.2f}")
        print(f"  Parameter B: {row['Parameter_B']:.6f}")
        print(f"  Total steady-state collapse point: {row['Collapse_q_total']}")
        print(f"  Import steady-state collapse point: {row['Collapse_q_in']}")
        print(f"  Export steady-state collapse point: {row['Collapse_q_out']}")
        print()
else:
    print("No collapse points detected, cannot plot change curve")

print("\nAll years analysis completed!")