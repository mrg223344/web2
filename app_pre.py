import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# 设置页面标题
st.set_page_config(page_title="乳腺癌预测模型", layout="wide")
st.title("乳腺癌预测模型")


# 定义MultiHeadSelfAttention类
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        # 生成QKV
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        # 合并多头
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        x = self.proj(x)
        return x


# 定义AutoInt模型
class AutoInt(nn.Module):
    def __init__(self, num_cont, cat_dims, embed_dim=32, num_heads=4, num_layers=3):
        super().__init__()
        # 连续特征处理
        self.cont_embed = nn.Linear(num_cont, embed_dim)
        self.cont_bn = nn.BatchNorm1d(num_cont)
        # 分类特征嵌入
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in cat_dims
        ])
        # 注意力交互层
        self.interaction_layers = nn.ModuleList([
            MultiHeadSelfAttention(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        # 输出层
        self.output = nn.Sequential(
            nn.Linear((len(cat_dims) + 1) * embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, cat, cont):
        # 连续特征处理
        cont = self.cont_bn(cont)
        cont_embed = self.cont_embed(cont).unsqueeze(1)  # [bs, 1, dim]
        # 分类特征嵌入
        cat_embeds = [embed(cat[:, i]) for i, embed in enumerate(self.cat_embeds)]
        cat_embeds = torch.stack(cat_embeds, dim=1)  # [bs, num_cat, dim]
        # 合并特征
        x = torch.cat([cont_embed, cat_embeds], dim=1)  # [bs, num_features, dim]
        # 多层交互
        for layer in self.interaction_layers:
            x = layer(x) + x  # 残差连接
        # 展平输出
        x = x.flatten(start_dim=1)
        return self.output(x)


# 加载模型
@st.cache_resource
def load_model():
    # 定义类别维度 - 根据您的数据
    categories = [3, 4, 4, 2, 3, 4, 2, 4, 4, 4, 6, 6, 2, 2, 2, 2, 4]

    # 初始化模型
    model = AutoInt(
        num_cont=3,
        cat_dims=categories,
        embed_dim=32,
        num_heads=4,
        num_layers=3
    )

    # 加载预训练权重
    try:
        model.load_state_dict(torch.load('e:\\website\\web2\\model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except:
        st.error("模型加载失败，请确保模型文件存在于正确路径")
        return None


# 加载标准化器
@st.cache_resource
def load_scaler():
    try:
        import joblib
        return joblib.load('e:\\website\\web2\\scaler2.joblib')
    except:
        st.warning("标准化器加载失败，将使用默认标准化")
        return None


# 创建侧边栏输入
st.sidebar.header("输入患者信息")

# 连续变量
st.sidebar.subheader("连续变量")
age = st.sidebar.slider("年龄", min_value=19, max_value=79, value=50, help="范围: 19-79岁")
tumor_size = st.sidebar.slider("肿瘤大小（毫米，以最大直径测量）", min_value=1, max_value=150, value=20,
                               help="范围: 1-150")
time_to_response = st.sidebar.slider("确诊到治疗响应时间（天）", min_value=0, max_value=731, value=30,
                                     help="范围: 0-731天")

# 分类变量
st.sidebar.subheader("分类变量")
race = st.sidebar.selectbox("种族",
                            options=[1, 2, 3],
                            format_func=lambda x:
                            {1: "白种人", 2: "黑种人", 3: "亚裔、太平洋岛屿原住民、美洲印第安人/阿拉斯加原住民"}[x])

marital_status = st.sidebar.selectbox("婚姻状况",
                                      options=[1, 2, 3, 4],
                                      format_func=lambda x:
                                      {1: "已婚/有稳定伴侣", 2: "离婚/分居", 3: "丧偶", 4: "从未结婚"}[x])

family_income = st.sidebar.selectbox("家庭中位收入（美元）",
                                     options=[1, 2, 3, 4],
                                     format_func=lambda x:
                                     {1: "0-60000", 2: "60000-80000", 3: "80000-100000", 4: "100000以上"}[x])

laterality = st.sidebar.selectbox("侧别性",
                                  options=[1, 2],
                                  format_func=lambda x: {1: "左侧", 2: "右侧"}[x])

histology = st.sidebar.selectbox("组织学类型",
                                 options=[1, 2, 3],
                                 format_func=lambda x:
                                 {1: "导管癌 (8500-8508)", 2: "小叶癌 (8520-8524)", 3: "其他特殊类型"}[x])

grade = st.sidebar.selectbox("组织学分级",
                             options=[1, 2, 3, 4],
                             format_func=lambda x:
                             {1: "高分化；I级", 2: "中分化；II级", 3: "低分化；III级", 4: "未分化；IV级"}[x])

pr_status = st.sidebar.selectbox("PR状态",
                                 options=[0, 1],
                                 format_func=lambda x: {0: "阴性", 1: "阳性"}[x])

breast_subtype = st.sidebar.selectbox("乳腺亚型",
                                      options=[1, 2, 3, 4],
                                      format_func=lambda x:
                                      {1: "HR-/HER2-", 2: "HR-/HER2+", 3: "HR+/HER2-", 4: "HR+/HER2+"}[x])

t_stage = st.sidebar.selectbox("T分期",
                               options=[1, 2, 3, 4],
                               format_func=lambda x: {1: "T1期", 2: "T2期", 3: "T3期", 4: "T4期"}[x])

n_stage = st.sidebar.selectbox("N分期",
                               options=[1, 2, 3, 4],
                               format_func=lambda x: {1: "N0期", 2: "N1期", 3: "N2期", 4: "N3期"}[x])

lymph_nodes_examined = st.sidebar.selectbox("局部淋巴结检出数",
                                            options=[0, 1, 2, 3, 4, 5],
                                            format_func=lambda x: "5及以上" if x == 5 else str(x))

lymph_nodes_positive = st.sidebar.selectbox("局部淋巴结阳性数",
                                            options=[0, 1, 2, 3, 4, 5],
                                            format_func=lambda x: "5及以上" if x == 5 else str(x))

liver_metastasis = st.sidebar.selectbox("肝转移",
                                        options=[0, 1],
                                        format_func=lambda x: {0: "否", 1: "是"}[x])

lung_metastasis = st.sidebar.selectbox("肺转移",
                                       options=[0, 1],
                                       format_func=lambda x: {0: "否", 1: "是"}[x])

radiation = st.sidebar.selectbox("放疗",
                                 options=[0, 1],
                                 format_func=lambda x: {0: "否", 1: "是"}[x])

chemotherapy = st.sidebar.selectbox("化疗",
                                    options=[0, 1],
                                    format_func=lambda x: {0: "否", 1: "是"}[x])

surgery = st.sidebar.selectbox("原发部位手术",
                               options=[1, 2, 3, 4],
                               format_func=lambda x: {1: "未进行手术（00）",
                                                      2: "局部切除/破坏性手术（20-30）",
                                                      3: "乳房切除术（40-59）",
                                                      4: "其他或未明确手术(60-80)"}[x])

# 预测按钮
predict_button = st.sidebar.button("预测")

# 主要内容区域
st.write("### 模型说明")
st.write("本模型基于AutoInt深度学习架构，用于预测乳腺癌患者的预后情况。")
st.write("请在左侧输入患者的相关信息，然后点击'预测'按钮获取结果。")

# 加载模型
model = load_model()
scaler = load_scaler()


# 预测函数
def predict(model, continuous_data, categorical_data):
    # 标准化连续变量
    if scaler is not None:
        continuous_data = scaler.transform(continuous_data.reshape(1, -1))
    else:
        # 简单标准化
        continuous_data = (continuous_data - np.array([50, 20, 30])) / np.array([15, 30, 100])

    # 转换为张量
    cat_tensor = torch.tensor(categorical_data, dtype=torch.long).unsqueeze(0)
    cont_tensor = torch.tensor(continuous_data, dtype=torch.float)

    # 预测
    with torch.no_grad():
        output = model(cat_tensor, cont_tensor).squeeze()
        prob = torch.sigmoid(output).item()
        label = 1 if prob > 0.5 else 0

    return prob, label


# 当点击预测按钮时
if predict_button:
    if model is not None:
        # 准备数据
        continuous_data = np.array([age, tumor_size, time_to_response])
        categorical_data = np.array([
            race - 1,  # 调整为从0开始的索引
            marital_status - 1,
            family_income - 1,
            laterality - 1,
            histology - 1,
            grade - 1,
            pr_status,
            breast_subtype - 1,
            t_stage - 1,
            n_stage - 1,
            lymph_nodes_examined,
            lymph_nodes_positive,
            liver_metastasis,
            lung_metastasis,
            radiation,
            chemotherapy,
            surgery - 1
        ])

        try:
            # 预测
            prob, label = predict(model, continuous_data, categorical_data)

            # 显示结果
            st.write("## 预测结果")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("输出概率", f"{prob:.4f}")

            with col2:
                result_text = "阳性" if label == 1 else "阴性"
                st.metric("预测结果", result_text)

            # 结果解释
            st.write("### 结果解释")
            if label == 1:
                st.write("模型预测该患者为阳性，建议进一步临床评估。")
            else:
                st.write("模型预测该患者为阴性，但请结合临床情况综合判断。")

            st.write("注意：本模型仅作为辅助工具，不能替代专业医生的诊断。")

        except Exception as e:
            st.error(f"预测过程中出现错误: {e}")
    else:
        st.error("模型未成功加载，无法进行预测")
