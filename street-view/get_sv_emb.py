import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


# 引入您提供的模型定义
# 确保此脚本放在 MuseCL-main/street-view/ 目录下，否则 import 路径需调整
from model.inceptionv3 import PlaceImageSkipGram

# ================= 配置区域 =================
# 1. 权重文件路径 (请修改为您训练 sv_train.py 后生成的 .tar 文件)
# 通常在 street-view/ 目录或您指定的 save_dir 下
ckpt_path = r"D:\work\pyProject\MuseCL-main\street-view\NY_SV_128_1_last.tar"  # <--- 请修改为您的实际权重文件名

# 2. 街景图片根目录
data_root = r"D:\work\pyProject\MuseCL-main\street_view_images"

# 3. 区域 ID 列表文件
region_idx_path = r"D:\work\pyProject\MuseCL-main\street_view_sample\region_idx.npy"

# 4. 输出路径
output_path = r"D:\work\pyProject\MuseCL-main\prediction-tasks\emb\sv_embedding.npy"

# 5. 嵌入维度 (必须与训练时一致，默认128)
EMBEDDING_DIM = 144


# ===========================================

def load_model(ckpt_path, device):
    """加载训练好的 InceptionV3 模型"""
    model = PlaceImageSkipGram(embedding_dim=EMBEDDING_DIM)

    if os.path.exists(ckpt_path):
        print(f"正在加载权重: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        # 处理 .tar 字典结构
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"警告: 找不到权重文件 {ckpt_path}，将使用随机初始化！")

    model = model.to(device)
    model.eval()
    return model


class RegionStreetViewDataset(Dataset):
    """
    专门用于推理的 Dataset
    给定一个 region_id，加载该文件夹下所有图片
    """

    def __init__(self, region_id, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []

        # 构造文件夹路径: manhattan_streetview_images/region_{id}
        # 注意：这里根据您的描述是 region_{i}
        region_dir = os.path.join(root_dir, f"region_{region_id}")

        if os.path.exists(region_dir):
            # 获取所有图片 (过滤隐藏文件)
            files = [f for f in os.listdir(region_dir) if not f.startswith('.')]
            self.image_paths = [os.path.join(region_dir, f) for f in files]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # 返回全黑图片防止崩溃
            return torch.zeros(3, 299, 299)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 准备数据预处理 (必须与 sv_train.py 中的 'val' 保持一致)
    # InceptionV3 标准输入是 299x299
    data_transforms = transforms.Compose([
        transforms.Resize((299, 299)),  # 强制缩放
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 2. 加载模型
    model = load_model(ckpt_path, device)

    # 3. 加载区域 ID
    region_idx = np.load(region_idx_path).tolist()
    # 确保排序，与后续任务对齐
    region_idx.sort(key=lambda x: int(x) if x.isdigit() else x)

    print(f"共需处理 {len(region_idx)} 个区域")

    final_embeddings = []

    # 4. 逐个区域处理
    with torch.no_grad():
        for rid in tqdm(region_idx):
            # 为每个区域创建一个临时 Dataset，加载该区域所有图片
            dataset = RegionStreetViewDataset(rid, data_root, transform=data_transforms)

            if len(dataset) == 0:
                # 如果该区域没有图片，填入全0向量
                # print(f"警告: 区域 {rid} 下没有图片")
                final_embeddings.append(np.zeros(EMBEDDING_DIM))
                continue

            # 使用 DataLoader 批量推理该区域的图片
            # batch_size 可以设大一点，因为只是推理
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

            region_feats = []
            for imgs in loader:
                imgs = imgs.to(device)
                feats = model(imgs)  # (Batch, 128)
                region_feats.append(feats.cpu().numpy())

            # 将所有 batch 的特征拼起来: (Total_Images, 128)
            region_feats = np.concatenate(region_feats, axis=0)

            # 【核心步骤】取平均值 (Mean Pooling) 作为该区域的最终嵌入
            avg_feat = np.mean(region_feats, axis=0)

            final_embeddings.append(avg_feat)

    # 5. 保存
    final_embeddings = np.array(final_embeddings)

    # 检查保存目录是否存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.save(output_path, final_embeddings)
    print(f"街景嵌入已保存至: {output_path}")
    print(f"最终形状: {final_embeddings.shape}")