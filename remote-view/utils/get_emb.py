import argparse
import os
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision.transforms import transforms
from tqdm import tqdm
import numpy as np


# --- 1. 定义与训练时一致的模型结构 ---
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_resnet18(output_dim, ckpt_path):
    # 必须与训练时结构一致：pretrained=True
    model = torchvision.models.resnet18(pretrained=True)

    # 修改全连接层结构
    model.fc = Identity()

    # --- 加载训练好的权重 ---
    print(f"正在加载权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path)

    # 提取模型权重部分 ('model_state_dict')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint  # 兼容直接保存 state_dict 的情况

    # 删除全连接层的旧权重（因为我们要重新初始化或它们不匹配）
    # 注意：我们在训练时已经训练了新的 FC 层，所以这里加载逻辑要小心
    # 如果 checkpoint 包含 fc.weight，我们应该保留它！
    # 但 MuseCL 原代码里有 pop 操作，这是为了防止维度不匹配。
    # 既然这是你自己训练的，维度肯定匹配，可以直接加载。

    # 修正加载逻辑：过滤掉不匹配的键（以防万一），但通常你训练的就是 128 维，直接加载即可
    model.load_state_dict(state_dict, strict=False)

    # 重新定义 FC 层 (确保与训练时 output_dim 一致)
    # 注意：如果 load_state_dict 成功加载了 fc 参数，这里重新定义会覆盖吗？
    # 正确的做法是先定义结构，再加载参数。
    model.fc = torch.nn.Linear(512, output_dim)

    # 再次加载一次以确保 FC 层参数也被加载（如果 checkpoint 里有的话）
    model.load_state_dict(state_dict, strict=False)

    return model


# --- 2. 主执行逻辑 ---
if __name__ == "__main__":
    # 配置路径 (请根据实际情况修改)
    # 训练生成的权重文件，通常在 training_logs 文件夹里，找 epoch 数最大的那个 .tar
    ckpt_path = "../NY_RV_128_triplet_188_last.tar"  # <--- 请修改为你实际生成的 .tar 文件名

    # 图像文件夹
    data_path = "../../satellite_images"

    # 区域 ID 列表
    region_idx_path = "../../satellite_sample/region_idx.npy"

    # 输出保存路径
    output_npy_path = "../../prediction-tasks/emb/si_embedding.npy"  # 建议保存到这里供后续使用

    # ------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理 (与验证集保持一致: Resize -> CenterCrop -> ToTensor)
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        # 如果训练时加了 Normalize，这里也要加，原代码 rv_train 似乎只有 ToTensor，所以这里保持一致
    ])

    # 初始化模型
    model = get_resnet18(output_dim=144, ckpt_path=ckpt_path)
    model = model.to(device)
    model.eval()  # 开启评估模式

    # 加载区域 ID
    image_list = np.load(region_idx_path).tolist()
    # image_list 应该是 ['0', '1', '2' ...]

    # 确保排序，保证嵌入向量的行索引与 ID 对应
    # 如果列表里的元素是数字字符串，按数字排序更安全
    image_list.sort(key=lambda x: int(x) if x.isdigit() else x)

    emb = []
    print("开始生成嵌入...")

    with torch.no_grad():  # 不计算梯度，节省显存
        for region_id in tqdm(image_list):
            # 构造文件名：region0.png
            image_name = f"region{region_id}.png"
            image_path = os.path.join(data_path, image_name)

            try:
                image = Image.open(image_path).convert('RGB')
                image = data_transforms(image)
                image = image.unsqueeze(0).to(device)  # 增加 batch 维度: [1, 3, 256, 256]

                image_emb = model(image)  # 输出 [1, 128]
                emb.append(image_emb.cpu().numpy()[0])  # 转回 numpy 并存入列表

            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                # 如果出错，填入全0向量占位，防止后续索引错位
                emb.append(np.zeros(128))

    # 保存最终结果
    if not os.path.exists(os.path.dirname(output_npy_path)):
        os.makedirs(os.path.dirname(output_npy_path))

    np.save(output_npy_path, np.array(emb))
    print(f"嵌入已保存至: {output_npy_path}")
    print(f"最终形状: {np.array(emb).shape}")