import heapq
import os
import csv
import numpy as np
from random import choice
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 加载相似度矩阵 (请确保文件名与您生成的一致)
similarity_path = "../similarity/similarity_POI_vsgre.npy"  # 或者 "similarity_Mobility_vsgre.npy"
similarity = np.load(similarity_path)

# 确保转换为 list 格式方便处理
if isinstance(similarity, np.ndarray):
    similarity = similarity.tolist()

# 2. 加载区域 ID 列表
# (必须是之前生成的只包含数字字符串的列表，如 ['0', '1', '2'...])
region_idx_path = "region_idx.npy"
region_idx = np.load(region_idx_path).tolist()

# 3. 输出的 CSV 文件名
output_csv_name = "train_pairs_satellite.csv"

# ================= 主逻辑 =================

a = []  # 用于存储 [anchor, positive, negative] 路径

print("正在生成对比样本对...")

for num in tqdm(range(10000)):  # 生成 10000 对样本

    # --- 1. 确定 Anchor (锚点) ---
    anc_region = choice(region_idx)  # 随机选一个区域ID，例如 '0'

    # 【修改点】：直接构造文件名，不再去文件夹遍历
    # 假设您的图片在 root_dir 下，文件名为 region0.png
    anc_image_name = f"region{anc_region}.png"
    anc_path = anc_image_name

    # --- 2. 确定 Positive (正样本) ---
    # 取出该区域对应的相似度向量
    anc_similarity = np.array(similarity[int(anc_region)])

    # 找出相似度最高的 2 个索引 (索引 0 通常是它自己，索引 1 是最相似的邻居)
    # 注意：这里 range 的长度要和相似度矩阵的列数一致
    top_simi = heapq.nlargest(2, range(len(anc_similarity)), anc_similarity.take)

    # 【修改点】：直接取 top_simi[1] 作为正样本，符合论文"相似度最高"的定义
    # 原代码是 choice(top_simi[1:])，因为我们只取了 top 2，所以效果是一样的
    pos_region_idx = top_simi[1]
    pos_region = str(pos_region_idx)

    # 构造正样本路径
    pos_image_name = f"region{pos_region}.png"
    pos_path = pos_image_name

    # --- 3. 确定 Negative (负样本) ---
    neg_region = choice(region_idx)
    # 确保负样本不等于 anchor 且不等于 positive
    while neg_region == anc_region or neg_region == pos_region:
        neg_region = choice(region_idx)

    # 构造负样本路径
    neg_image_name = f"region{neg_region}.png"
    neg_path = neg_image_name

    # --- 4. 添加到列表 ---
    # 最终 dataset 读取时会用 root_dir + anc_path 拼接
    line = [anc_path, pos_path, neg_path]
    a.append(line)

# ================= 保存结果 =================
print(f"样本生成完毕，正在写入 {output_csv_name} ...")

# newline='' 防止在 Windows 下产生空行
with open(output_csv_name, "w", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(a)

print("完成！")