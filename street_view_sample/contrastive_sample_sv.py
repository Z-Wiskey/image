import heapq
import os
import csv
import numpy as np
from random import choice
from tqdm import tqdm

# ==================== 配置区域 ====================

# 1. 图片根目录 (您的街景文件夹)
# 请确保这里填的是包含 region_0, region_1... 这些文件夹的父目录
image_root_dir = "manhattan_streetview_images"

# 2. 相似度矩阵 (街景必须用 Mobility 相似度)
similarity_path = "../similarity/similarity_Mobility_vsgre.npy"
similarity = np.load(similarity_path)
if isinstance(similarity, np.ndarray):
    similarity = similarity.tolist()

# 3. 区域 ID 列表 (纯数字 ID，如 ['0', '1', ...])
region_idx_path = "region_idx.npy"
region_idx = np.load(region_idx_path).tolist()

# 4. 输出文件名
train_csv_name = "train_pairs_street_view.csv"
val_csv_name = "val_pairs_street_view.csv"

# 5. 采样数量
NUM_TRAIN = 50000
NUM_VAL = 10000


# ==================== 辅助函数 ====================

def get_random_image_from_region(root_dir, region_id):
    """
    从 region_{id} 文件夹中随机选择一张图片，返回相对路径
    例如: region_0/pt1_h90.jpg
    """
    # 构造文件夹名: region_0
    region_folder_name = f"region_{region_id}"
    full_folder_path = os.path.join(root_dir, region_folder_name)

    # 检查文件夹是否存在
    if not os.path.exists(full_folder_path):
        return None

    # 获取该文件夹下所有以 pt 开头的文件 (根据您的描述 pt{i}_h{90})
    # 过滤掉隐藏文件
    images = [f for f in os.listdir(full_folder_path) if f.startswith('pt') and not f.startswith('.')]

    if len(images) == 0:
        return None

    # 随机选一张
    chosen_img = choice(images)

    # 返回相对路径: region_{i}/image_name
    return os.path.join(region_folder_name, chosen_img)


def generate_samples(num_samples, desc="Generating"):
    """
    生成指定数量的样本列表
    """
    samples = []
    # 使用 tqdm 显示进度条
    pbar = tqdm(total=num_samples, desc=desc)

    while len(samples) < num_samples:
        # --- 1. Anchor ---
        anc_id = choice(region_idx)  # 数字 ID，如 '0'
        anc_path = get_random_image_from_region(image_root_dir, anc_id)

        # 如果该区域文件夹为空或不存在，重新选
        if anc_path is None:
            continue

        # --- 2. Positive (相似度最高) ---
        anc_sim_vec = np.array(similarity[int(anc_id)])
        # 取 Top-2 (第0个是自己，第1个是最相似的邻居)
        top_simi_indices = heapq.nlargest(2, range(len(anc_sim_vec)), anc_sim_vec.take)

        pos_id = str(top_simi_indices[1])  # 获取邻居 ID
        pos_path = get_random_image_from_region(image_root_dir, pos_id)

        if pos_path is None:
            continue

        # --- 3. Negative (随机不相似) ---
        neg_id = choice(region_idx)
        while neg_id == anc_id or neg_id == pos_id:
            neg_id = choice(region_idx)

        neg_path = get_random_image_from_region(image_root_dir, neg_id)

        if neg_path is None:
            continue

        # --- 4. 添加记录 ---
        # 路径格式示例: region_0/pt1_h90.jpg
        samples.append([anc_path, pos_path, neg_path])
        pbar.update(1)

    pbar.close()
    return samples


def save_to_csv(filename, data):
    print(f"正在写入 {filename} ...")
    with open(filename, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    print("写入完成。")


# ==================== 主逻辑 ====================

if __name__ == "__main__":
    print(f"开始处理街景数据，根目录: {image_root_dir}")

    # 生成训练集
    train_data = generate_samples(NUM_TRAIN, desc="训练集采样")
    save_to_csv(train_csv_name, train_data)

    # 生成验证集
    val_data = generate_samples(NUM_VAL, desc="测试集采样")
    save_to_csv(val_csv_name, val_data)

    print("\n全部完成！请在 sv_train.py 中使用这两个 CSV 文件。")