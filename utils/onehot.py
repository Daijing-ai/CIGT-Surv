import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. 自动统计每个特征的类别（自适应）
# ---------------------------------------------------------
def build_category_maps(df, categorical_cols):
    category_maps = {}
    for col in categorical_cols:
        cats = sorted(df[col].dropna().unique().tolist())
        category_maps[col] = cats
    return category_maps


# ---------------------------------------------------------
# 2. 对单个样本编码成 50×7 的矩阵
# ---------------------------------------------------------
def encode_row(row, category_maps):
    """
    row: DataFrame 的一行
    category_maps: 每个特征 -> 类别列表
    返回: 50×7 的矩阵
    """
    num_features = len(category_maps) + 1   # +1 for age
    mat = np.zeros((50, num_features))      # 固定 50×7

    col_idx = 0

    # 1. 分类特征 one-hot（stage, T, N, M, gender, race）
    for feature, cats in category_maps.items():
        value = row[feature]
        if value in cats:
            row_idx = cats.index(value)
            # 防止某个特征类别数 > 50，直接截断到最后一行
            row_idx = min(row_idx, 49)
            mat[row_idx, col_idx] = 1
        col_idx += 1

    # 2. 年龄编码：每 2 岁一个年龄段
    #    0–2 -> 0, 2–4 -> 1, ..., 98–100 -> 49, >100 -> 49
    age = row["age"]
    if pd.notnull(age):
        age_bin = int(age // 2)
        age_bin = max(0, min(age_bin, 49))
        mat[age_bin, col_idx] = 1

    return mat


# ---------------------------------------------------------
# 3. 主函数：对整个临床表进行编码
# ---------------------------------------------------------
def encode_clinical_dataframe(df):
    """
    输入: 含有 TCGA_BLCA 临床信息的 DataFrame
    输出:
        encoded_array: (样本数, 50, 7)
        category_maps: 每个分类特征的类别映射
    """
    categorical_cols = [
        "ajcc_pathologic_stage",
        "ajcc_pathologic_t",
        "ajcc_pathologic_n",
        "ajcc_pathologic_m",
        "gender",
        "race"
    ]
    # censorship, survival_months 不编码

    # 1. 自动生成类别映射
    category_maps = build_category_maps(df, categorical_cols)

    # 2. 编码所有样本
    encoded = []
    for _, row in df.iterrows():
        mat = encode_row(row, category_maps)
        encoded.append(mat)

    return np.array(encoded), category_maps


# df = pd.read_csv("/home/guestdj/nvme3n1/DJ/Survival-Main/clinical_csv/TCGA_KIRP.csv")

# encoded_array, category_maps = encode_clinical_dataframe(df)

# print(encoded_array.shape)   # (样本数, 50, 7)
# print(category_maps)
