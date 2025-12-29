import numpy as np
from scipy.spatial.distance import pdist, squareform


def calculate_log_gaussian_similarity(data_matrix):
    """
    根据论文 4.2.1 节计算结构相似度 (S_struct)
    适用于 POI 矩阵 (P) 和 人类流动矩阵 (M)

    参数:
        data_matrix: shape (N, D), N为区域数量, D为特征维度
                     - 对于 POI: D = POI类型数量
                     - 对于 Mobility: D = 区域数量 (N)
    返回:
        similarity_matrix: shape (N, N), 范围 [0, 1]
    """

    # ---------------------------------------------------------
    # 步骤 1: 对数变换 (公式 1 的一部分)
    # ln(u_i + 1)
    # ---------------------------------------------------------
    # 加上 1.0 防止 log(0)
    data_log = np.log(data_matrix + 1.0)

    # ---------------------------------------------------------
    # 步骤 2: 计算两两之间的欧氏距离 (公式 1)
    # Dist_{i,j} = || ln(u_i + 1) - ln(u_j + 1) ||_2
    # ---------------------------------------------------------
    # pdist 计算压缩后的距离矩阵，metric='euclidean' 即 L2 范数
    pairwise_dists = pdist(data_log, metric='euclidean')

    # ---------------------------------------------------------
    # 步骤 3: 计算高斯核带宽参数 sigma (公式 2 的说明)
    # sigma 设置为所有区域对距离的中位数
    # ---------------------------------------------------------
    sigma = np.median(pairwise_dists)

    # 防止 sigma 为 0 (如果所有数据都一样)
    if sigma == 0:
        sigma = 1.0

    # ---------------------------------------------------------
    # 步骤 4: 计算结构相似度 (公式 2)
    # S_{struct}(r_i, r_j) = exp( - Dist_{i,j}^2 / (2 * sigma^2) )
    # ---------------------------------------------------------
    # 将距离平方
    dists_squared = pairwise_dists ** 2

    # 应用高斯核
    # 注意：pdist 返回的是压缩数组，直接操作即可
    similarity_compressed = np.exp(- dists_squared / (2 * (sigma ** 2)))

    # 将压缩形式转换为 N x N 的方阵
    similarity_matrix = squareform(similarity_compressed)

    # 对角线元素（自己和自己）的距离是 0，相似度应为 exp(0) = 1
    # squareform 会把对角线设为 0，我们需要修正为 1
    np.fill_diagonal(similarity_matrix, 1.0)

    return similarity_matrix


# =========================================================
# 使用示例：生成 MuseCL 需要的 .npy 文件
# =========================================================

# 1. 加载你的原始数据
# 假设 POI_data 是 (区域数 N, POI类型数 K) 的矩阵
# 假设 Mobility_data 是 (区域数 N, 区域数 N) 的矩阵 (对应论文中 m_i 是 1xN 向量)
# 你需要根据实际情况加载你的 .npy 数据

# 示例：加载假设的数据路径
poi_vectors = np.load("./poi_dist.npy")
mobility_vectors = np.load("./mobility_adj.npy")

mobility_vectors = mobility_vectors.squeeze()
print(poi_vectors.shape)
print(mobility_vectors.shape)

# 为了演示，这里生成随机数据
# N_REGIONS = 180  # 假设你有180个区域
# N_POI_TYPES = 14
# poi_vectors = np.random.randint(0, 100, (N_REGIONS, N_POI_TYPES)).astype(float)
# mobility_vectors = np.random.randint(0, 500, (N_REGIONS, N_REGIONS)).astype(float)

print("正在计算 POI 相似度...")
S_poi = calculate_log_gaussian_similarity(poi_vectors)
# 保存用于卫星图像训练的相似度
np.save("./similarity_POI_vsgre.npy", S_poi)
print(f"POI 相似度矩阵已保存，形状: {S_poi.shape}")

print("正在计算 Mobility 相似度...")
S_mob = calculate_log_gaussian_similarity(mobility_vectors)
# 保存用于街景图像训练的相似度
np.save("./similarity_Mobility_vsgre.npy", S_mob)
print(f"Mobility 相似度矩阵已保存，形状: {S_mob.shape}")

# ---------------------------------------------------------
# 补充：对 contrastive_sample_sv.py 的微调建议
# ---------------------------------------------------------
