# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:39:37 2026

@author: mengx
"""

import pandas as pd
import numpy as np
import os
import networkx as nx
import chardet
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

# 设置文件路径
base_path = r"E:\论文写作\35 TRE"
raw_data_path = os.path.join(base_path, "2 每年原始数据\\1 原数据")
output_path = os.path.join(base_path, "2 每年原始数据\\3 网络数据_新")
topology_path = os.path.join(base_path, "2 每年原始数据\\4 网络数据_作图")

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)
os.makedirs(topology_path, exist_ok=True)

# 检测文件编码的函数
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read(10000)
        result = chardet.detect(rawdata)
        return result['encoding']

# 检测两个文件的编码
file1 = os.path.join(raw_data_path, "【import2014_2024】TradeData_7_29_2026_22_18_48.csv")
file2 = os.path.join(raw_data_path, "【import2006_2013】TradeData_7_29_2026_22_19_41.csv")
file3 = os.path.join(raw_data_path, "【import1994_2006】TradeData_7_29_2026_22_21_4.csv")

# 获取文件编码
encoding1 = detect_encoding(file1) or 'gbk'
encoding2 = detect_encoding(file2) or 'gbk'
encoding3 = detect_encoding(file3) or 'gbk'

print(f"文件1编码: {encoding1}")
print(f"文件2编码: {encoding2}")
print(f"文件3编码: {encoding3}")

# 读取CSV文件，使用检测到的编码
try:
    df1 = pd.read_csv(file1, encoding=encoding1, index_col=False)
except:
    print(f"使用{encoding1}读取文件1失败，尝试使用gbk")
    df1 = pd.read_csv(file1, encoding='gbk', index_col=False)

try:
    df2 = pd.read_csv(file2, encoding=encoding2, index_col=False)
except:
    print(f"使用{encoding2}读取文件2失败，尝试使用gbk")
    df2 = pd.read_csv(file2, encoding='gbk', index_col=False)
    
try:
    df3 = pd.read_csv(file3, encoding=encoding2, index_col=False)
except:
    print(f"使用{encoding3}读取文件2失败，尝试使用gbk")
    df3 = pd.read_csv(file3, encoding='gbk', index_col=False)

# 合并数据
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# 从数据中排除"World"条目`
combined_df = combined_df[(combined_df['reporterISO'] != 'World') & 
                          (combined_df['partnerISO'] != 'World') & 
                          (combined_df['partnerISO'] != 0)]
#reporterDesc换成reporterISO
#partnerDesc换成partnerISO

# 提取所有国家列表（排除"World"）
all_reporters = set(combined_df['reporterISO'])
all_partners = set(combined_df['partnerISO'])
all_countries = sorted(all_reporters | all_partners)

# 确保"World"不在国家列表中
if 'World' in all_countries:
    all_countries.remove('World')
    print("已从国家列表中排除'World'")

print(f"共有 {len(all_countries)} 个国家和地区")

# 保存国家列表（按索引顺序）
country_index_path = os.path.join(output_path, "country_index.csv")
pd.DataFrame({
    'Index': range(len(all_countries)),
    'Country': list(all_countries)
}).to_csv(country_index_path, index=False, encoding='utf-8-sig')

print(f"国家索引已保存至: {country_index_path}")

# 只保留进口数据（flowDesc == 'Import'）
import_df = combined_df[combined_df['flowDesc'] == 'Import']

# 按年份分组
grouped = import_df.groupby('refYear')

# 创建国家索引映射
country_index = {country: idx for idx, country in enumerate(all_countries)}

# 初始化拓扑属性数据收集器
topology_data = []

# 处理每个年份的数据
for year, year_df in grouped:
    print(f"\n处理 {year} 年数据...")
    
    # 创建空矩阵
    matrix = np.zeros((len(all_countries), len(all_countries)))
    
    # 按出口国-进口国分组求和（处理多重边）
    grouped_edges = year_df.groupby(['partnerISO', 'reporterISO'])['primaryValue'].sum().reset_index()
    
    # 1. 保存边列表格式文件 (source, target, weight)
    edgelist_df = grouped_edges.copy()
    edgelist_df.columns = ['source', 'target', 'weight']
    
    # 从边列表中排除包含"World"的记录
    edgelist_df = edgelist_df[(edgelist_df['source'] != 'World') & 
                             (edgelist_df['target'] != 'World')]
    
    edgelist_path = os.path.join(output_path, f"edgelist_{year}.csv")
    edgelist_df.to_csv(edgelist_path, index=False, encoding='utf-8-sig')
    print(f"边列表已保存至: {edgelist_path} (包含 {len(edgelist_df)} 条边)")
    
    # 2. 创建邻接矩阵
    for _, row in grouped_edges.iterrows():
        try:
            source = country_index[row['partnerISO']]
            target = country_index[row['reporterISO']]
            matrix[source, target] = row['primaryValue']
        except KeyError:
            continue
    
    # 3. 保存邻接矩阵为Excel格式
    matrix_df = pd.DataFrame(matrix, 
                            index=all_countries, 
                            columns=all_countries)
    matrix_excel_path = os.path.join(output_path, f"adj_matrix_{year}.xlsx")
    matrix_df.to_excel(matrix_excel_path, engine='openpyxl')
    print(f"邻接矩阵已保存至: {matrix_excel_path} ({len(all_countries)}×{len(all_countries)} 矩阵)")
    
    # 4. 创建NetworkX图对象并计算拓扑属性
    G = nx.DiGraph()
    for _, row in edgelist_df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
    
    print(f"{year} 年网络包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")
    
    # 计算拓扑属性
    print("计算拓扑属性...")
    
    # a. 度分布
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    avg_in_degree = np.mean(in_degrees)
    avg_out_degree = np.mean(out_degrees)
    
    # b. 聚集系数
    clustering_coeffs = nx.clustering(G)
    avg_clustering = np.mean(list(clustering_coeffs.values()))
    
    # c. 总贸易量（总连边的权重和）
    total_trade = sum(edgelist_df['weight'])
    
    # d. 平均路径长度（仅针对强连通分量）
    if nx.is_strongly_connected(G):
        avg_path_length = nx.average_shortest_path_length(G)
    else:
        # 计算最大强连通分量的平均路径长度
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        subgraph = G.subgraph(largest_scc)
        avg_path_length = nx.average_shortest_path_length(subgraph)
        print(f"  网络不是强连通，使用最大强连通分量计算路径长度（包含 {len(largest_scc)} 个节点）")
    
    # e. 同配性（度相关性）
    # 对于有向图，计算入度-入度、出度-出度和入度-出度的相关性
    in_in_assort = nx.degree_assortativity_coefficient(G, x='in', y='in')
    out_out_assort = nx.degree_assortativity_coefficient(G, x='out', y='out')
    in_out_assort = nx.degree_assortativity_coefficient(G, x='in', y='out')
    
    # 收集拓扑属性
    topology_entry = {
        'year': year,
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'avg_in_degree': avg_in_degree,
        'avg_out_degree': avg_out_degree,
        'avg_clustering': avg_clustering,
        'total_trade': total_trade,
        'avg_path_length': avg_path_length,
        'in_in_assortativity': in_in_assort,
        'out_out_assortativity': out_out_assort,
        'in_out_assortativity': in_out_assort
    }
    topology_data.append(topology_entry)
    
    print(f"拓扑属性计算完成:")
    print(f"  平均入度: {avg_in_degree:.4f}, 平均出度: {avg_out_degree:.4f}")
    print(f"  平均聚集系数: {avg_clustering:.4f}")
    print(f"  总贸易量: {total_trade:.2f}")
    print(f"  平均路径长度: {avg_path_length:.4f}")
    print(f"  同配性: in-in={in_in_assort:.4f}, out-out={out_out_assort:.4f}, in-out={in_out_assort:.4f}")

# 保存拓扑属性数据
topology_df = pd.DataFrame(topology_data)
topology_csv_path = os.path.join(topology_path, "supply_chain_topology.csv")
topology_df.to_csv(topology_csv_path, index=False, encoding='utf-8-sig')
print(f"\n所有年份的拓扑属性已保存至: {topology_csv_path}")

# 设置字体和大小变量
fontsize = 24
font_family = 'Times New Roman'  # 新罗马字体

# 1. 平均度分布
plt.figure(figsize=(15, 9))
plt.subplot(2, 3, 1)
plt.plot(topology_df['year'], topology_df['avg_in_degree'], 'o-', label='Average In-degree')
plt.plot(topology_df['year'], topology_df['avg_out_degree'], 's-', label='Average Out-degree')
plt.xlabel('Year', fontsize=fontsize, fontfamily=font_family)
plt.ylabel('Average Degree', fontsize=fontsize, fontfamily=font_family)
plt.title('Average Degree Distribution', fontsize=fontsize, fontfamily=font_family)
plt.grid(True)
plt.legend(fontsize=fontsize-5, prop={'family': font_family})

# 2. 平均聚集系数
plt.subplot(2, 3, 2)
plt.plot(topology_df['year'], topology_df['avg_clustering'], 'o-')
plt.xlabel('Year', fontsize=fontsize, fontfamily=font_family)
plt.ylabel('Average Clustering Coefficient', fontsize=fontsize, fontfamily=font_family)
plt.title('Average Clustering Coefficient', fontsize=fontsize, fontfamily=font_family)
plt.grid(True)

# 3. 总贸易量
plt.subplot(2, 3, 3)
plt.plot(topology_df['year'], topology_df['total_trade'] / 1e9, 'o-')  # 除以10^9，单位变为十亿
plt.xlabel('Year', fontsize=fontsize, fontfamily=font_family)
plt.ylabel('Total Trade Volume (Billion USD)', fontsize=fontsize, fontfamily=font_family)
plt.title('Total Trade Volume', fontsize=fontsize, fontfamily=font_family)
plt.grid(True)

# 4. 平均路径长度
plt.subplot(2, 3, 4)
plt.plot(topology_df['year'], topology_df['avg_path_length'], 'o-')
plt.xlabel('Year', fontsize=fontsize, fontfamily=font_family)
plt.ylabel('Average Path Length', fontsize=fontsize, fontfamily=font_family)
plt.title('Average Path Length', fontsize=fontsize, fontfamily=font_family)
plt.grid(True)

# 5. 同配性
plt.subplot(2, 3, 5)
plt.plot(topology_df['year'], topology_df['in_in_assortativity'], 'o-', label='In-in')
plt.plot(topology_df['year'], topology_df['out_out_assortativity'], 's-', label='Out-out')
plt.plot(topology_df['year'], topology_df['in_out_assortativity'], '^-', label='In-out')
plt.xlabel('Year', fontsize=fontsize, fontfamily=font_family)
plt.ylabel('Assortativity Coefficient', fontsize=fontsize, fontfamily=font_family)
plt.title('Assortativity', fontsize=fontsize, fontfamily=font_family)
plt.grid(True)
plt.legend(fontsize=fontsize-5, prop={'family': font_family})

# 6. 网络规模
plt.subplot(2, 3, 6)
plt.plot(topology_df['year'], topology_df['nodes'], 'o-', label='Nodes')
plt.plot(topology_df['year'], topology_df['edges'], 's-', label='Edges')
plt.xlabel('Year', fontsize=fontsize, fontfamily=font_family)
plt.ylabel('Count', fontsize=fontsize, fontfamily=font_family)
plt.title('Network Size', fontsize=fontsize, fontfamily=font_family)
plt.grid(True)
plt.legend(fontsize=fontsize-5, prop={'family': font_family})

plt.tight_layout()
plot_path = os.path.join(topology_path, "topology_properties.png")
plt.savefig(plot_path, dpi=300)
print(f"Topology properties plot saved to: {plot_path}")

# 保存拓扑数据到Excel
topology_excel_path = os.path.join(topology_path, "supply_chain_topology.xlsx")
topology_df.to_excel(topology_excel_path, index=False, engine='openpyxl')
print(f"Topology data saved to Excel: {topology_excel_path}")

print("\nAll data processing completed!")
print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")