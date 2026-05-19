import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ---- 页面设置 ----
st.set_page_config(page_title="Cookie Cats A/B测试", layout="wide")
st.title("🎮 Cookie Cats 关卡难度 A/B 测试仪表盘")
st.markdown("实验：将关卡障碍从第30关推迟至第40关，评估对玩家留存与活跃度的影响。")

# ---- 读取数据 ----
@st.cache_data
def load_data():
    df = pd.read_csv('C:/Users/92717/Desktop/ab-test-cookie-cats/数据/cookie_cats.csv')
    # 缩尾处理
    lower = df['sum_gamerounds'].quantile(0.01)
    upper = df['sum_gamerounds'].quantile(0.99)
    df['sum_gamerounds_win'] = df['sum_gamerounds'].clip(lower=lower, upper=upper)
    return df

df = load_data()

# ---- 侧边栏：交互控制 ----
st.sidebar.header("⚙️ 分析设置")
alpha = st.sidebar.slider("显著性水平 α", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
metric_choice = st.sidebar.selectbox("选择要查看的指标", 
                                     ["7日留存率", "1日留存率", "游戏总局数（缩尾后）"])

# ---- 核心指标卡片 ----
col1, col2, col3 = st.columns(3)
retention_1_gate30 = df[df['version']=='gate_30']['retention_1'].mean() * 100
retention_1_gate40 = df[df['version']=='gate_40']['retention_1'].mean() * 100
retention_7_gate30 = df[df['version']=='gate_30']['retention_7'].mean() * 100
retention_7_gate40 = df[df['version']=='gate_40']['retention_7'].mean() * 100
avg_rounds_gate30 = df[df['version']=='gate_30']['sum_gamerounds_win'].mean()
avg_rounds_gate40 = df[df['version']=='gate_40']['sum_gamerounds_win'].mean()

col1.metric("1日留存 (gate_30)", f"{retention_1_gate30:.1f}%", delta=None)
col2.metric("7日留存 (gate_30)", f"{retention_7_gate30:.1f}%", delta=None)
col3.metric("平均局数 (gate_30)", f"{avg_rounds_gate30:.1f}")

col1.metric("1日留存 (gate_40)", f"{retention_1_gate40:.1f}%", delta=None)
col2.metric("7日留存 (gate_40)", f"{retention_7_gate40:.1f}%", delta=None)
col3.metric("平均局数 (gate_40)", f"{avg_rounds_gate40:.1f}")

# ---- 假设检验结果展示 ----
st.header("📈 假设检验结果")
st.markdown(f"当前显著性水平 **α = {alpha:.2f}**")

# 执行检验函数（根据选择）
def run_test(df, metric):
    if metric in ["7日留存率", "1日留存率"]:
        col = 'retention_7' if '7日' in metric else 'retention_1'
        contingency = pd.crosstab(df['version'], df[col])
        chi2, p, _, _ = chi2_contingency(contingency)
        return chi2, p, "卡方检验", contingency
    else:
        g30 = df[df['version']=='gate_30']['sum_gamerounds_win']
        g40 = df[df['version']=='gate_40']['sum_gamerounds_win']
        t, p = ttest_ind(g30, g40, equal_var=False)
        return t, p, "Welch's t检验", None

chi2_or_t, p_value, test_name, table = run_test(df, metric_choice)

# 显示结论
if p_value < alpha:
    st.success(f"✅ 结果显著 ({test_name}, p = {p_value:.4f}) —— 拒绝原假设，两组{metric_choice}存在统计学差异")
else:
    st.info(f"❌ 结果不显著 ({test_name}, p = {p_value:.4f}) —— 不能拒绝原假设，两组{metric_choice}无显著差异")

# 如果是留存率，展示列联表
if table is not None:
    st.subheader("列联表")
    st.dataframe(table)

# ---- 可视化 ----
st.header("📊 可视化对比")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("留存率对比")
    retention_means = df.groupby('version')[['retention_1', 'retention_7']].mean() * 100
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    retention_means.plot(kind='bar', ax=ax1, color=['#4CAF50', '#FF9800'], edgecolor='black', alpha=0.85)
    ax1.set_xticklabels(['Gate 30', 'Gate 40'], rotation=0)
    ax1.set_ylabel('留存率 (%)')
    ax1.set_title('1日 vs 7日留存率')
    ax1.grid(axis='y', alpha=0.3)
    for p in ax1.patches:
        ax1.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width()/2., p.get_height()), 
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    st.pyplot(fig1)

with col_right:
    st.subheader("游戏总局数分布 (缩尾后)")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    df.boxplot(column='sum_gamerounds_win', by='version', ax=ax2, patch_artist=True,
               boxprops=dict(facecolor='#64B5F6', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    ax2.set_xticklabels(['Gate 30', 'Gate 40'])
    ax2.set_title('游戏局数分布')
    ax2.set_ylabel('游戏局数')
    st.pyplot(fig2)

# ---- 页脚 ----
st.markdown("---")
st.caption("数据来源：Kaggle Mobile Games A/B Testing | 项目代码：[GitHub](https://github.com/CipherZeroWW/ab-test-cookie-cats)")