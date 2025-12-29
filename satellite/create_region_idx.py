import os
import numpy as np
import re

# 1. 您的图像文件夹路径
image_dir = "../satellite_images"  # 请替换为您实际的文件夹路径

# 2. 读取所有文件
files = os.listdir(image_dir)

region_idx_list = []

# 定义正则，匹配 "region" 开头，数字中间，".png" 结尾
# 这样能保证提取的是纯数字 ID
pattern = re.compile(r"region(\d+)\.png")

for f in files:
    match = pattern.match(f)
    if match:
        # 提取括号内的数字部分，例如 "0", "1", "14"
        region_id = match.group(1)
        region_idx_list.append(region_id)

# 3. 极其重要：按数字大小排序！
# 必须确保列表是 ['0', '1', '2', ..., '9', '10', ...]
# 而不是字符串排序的 ['0', '1', '10', '11', '2', ...]
region_idx_list.sort(key=lambda x: int(x))

print(f"检测到 {len(region_idx_list)} 个有效区域")
print(f"前 5 个 ID: {region_idx_list[:5]}")
print(f"后 5 个 ID: {region_idx_list[-5:]}")

# 4. 保存
np.save("region_idx.npy", np.array(region_idx_list))
print("region_idx.npy 已生成！")