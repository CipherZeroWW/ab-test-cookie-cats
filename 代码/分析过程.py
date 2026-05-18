import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体，避免图表乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 读取数据 ====================
df = pd.read_csv('C:/Users/92717/Desktop/ab-test-cookie-cats/数据/cookie_cats.csv')

print("========== 数据概览 ==========")
print("前5行数据：")
print(df.head())
print("\n数据信息：")
print(df.info())
print("\n分组人数：")
print(df['version'].value_counts())
print("\n各版本留存率均值：")
print(df.groupby('version')[['retention_1', 'retention_7']].mean())
print("\n各版本游戏总局数均值：")
print(df.groupby('version')['sum_gamerounds'].mean())

# ==================== 2. 卡方检验：1日留存率 ====================
print("\n\n========== 1日留存率卡方检验 ==========")
contingency_1 = pd.crosstab(df['version'], df['retention_1'])
print("1日留存列联表：")
print(contingency_1)

chi2_1, p_1, dof_1, expected_1 = chi2_contingency(contingency_1)
print(f"chi2 = {chi2_1:.4f}, p值 = {p_1:.4f}")

# ==================== 3. 卡方检验：7日留存率 ====================
print("\n\n========== 7日留存率卡方检验 ==========")
contingency_7 = pd.crosstab(df['version'], df['retention_7'])
print("7日留存列联表：")
print(contingency_7)

chi2_7, p_7, dof_7, expected_7 = chi2_contingency(contingency_7)
print(f"chi2 = {chi2_7:.4f}, p值 = {p_7:.4f}")

# ==================== 4. 游戏总局数 t 检验（含异常值处理） ====================
print("\n\n========== 游戏总局数描述统计与异常值处理 ==========")
print(df['sum_gamerounds'].describe())

# 缩尾处理
lower = df['sum_gamerounds'].quantile(0.01)
upper = df['sum_gamerounds'].quantile(0.99)

print(f"\n1%分位数: {lower}, 99%分位数: {upper}")
print(f"处理前 - 均值: {df['sum_gamerounds'].mean():.2f}, 标准差: {df['sum_gamerounds'].std():.2f}")

df['sum_gamerounds_win'] = df['sum_gamerounds'].clip(lower=lower, upper=upper)

print(f"处理后 - 均值: {df['sum_gamerounds_win'].mean():.2f}, 标准差: {df['sum_gamerounds_win'].std():.2f}")

# 拆分两组并执行 t 检验
gate30_win = df[df['version'] == 'gate_30']['sum_gamerounds_win']
gate40_win = df[df['version'] == 'gate_40']['sum_gamerounds_win']

t_stat, p_ttest = ttest_ind(gate30_win, gate40_win, equal_var=False)
print(f"\nt 检验结果：t = {t_stat:.4f}, p值 = {p_ttest:.4f}")
print(f"gate_30 平均局数（处理后） = {gate30_win.mean():.2f}")
print(f"gate_40 平均局数（处理后） = {gate40_win.mean():.2f}")

# ==================== 5. 可视化 ====================
print("\n\n========== 生成图表 ==========")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- 留存率对比柱状图 ---
retention_means = df.groupby('version')[['retention_1', 'retention_7']].mean() * 100
retention_means.plot(kind='bar', ax=axes[0], color=['#4CAF50', '#FF9800'], edgecolor='black', alpha=0.85)
axes[0].set_title('不同版本 1日 vs 7日留存率对比', fontsize=14, fontweight='bold')
axes[0].set_xlabel('实验分组', fontsize=12)
axes[0].set_ylabel('留存率 (%)', fontsize=12)
axes[0].set_xticklabels(['Gate 30 (对照)', 'Gate 40 (实验)'], rotation=0)
axes[0].legend(['1日留存', '7日留存'])
axes[0].grid(axis='y', alpha=0.3)

# 在柱状图上添加数值标签
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height():.1f}%',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

# --- 游戏总局数分布箱线图 ---
df.boxplot(column='sum_gamerounds_win', by='version', ax=axes[1],
           patch_artist=True,
           boxprops=dict(facecolor='#64B5F6', alpha=0.7),
           medianprops=dict(color='red', linewidth=2))
axes[1].set_title('游戏总局数分布对比（缩尾处理后）', fontsize=14, fontweight='bold')
axes[1].set_xlabel('实验分组', fontsize=12)
axes[1].set_ylabel('游戏总局数', fontsize=12)
axes[1].set_xticklabels(['Gate 30 (对照)', 'Gate 40 (实验)'])

plt.tight_layout()
plt.savefig('C:/Users/92717/Desktop/ab-test-cookie-cats/图表/A_B测试可视化结果.png', dpi=300, bbox_inches='tight')
print("图表已保存到 图表/A_B测试可视化结果.png")