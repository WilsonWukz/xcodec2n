import random
import os

# 1. 配置路径
# 源文件：原本的全量训练列表
source_file = './data/output/train_clean_100_file_list.tsv'

# 目标文件：切分后的训练集 (80%)
train_output = './data/output/train_split_8.tsv'
# 目标文件：切分后的验证集 (20%)
val_output = './data/output/val_split_2.tsv'


# 2. 读取并处理
def split_dataset():
    if not os.path.exists(source_file):
        print(f"错误：找不到源文件 {source_file}")
        return

    print(f"正在读取 {source_file} ...")
    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 第一行通常是根目录路径（root_dir），需要保留，不参与 shuffle
    root_header = lines[0]
    data_lines = lines[1:]

    # 随机打乱
    random.shuffle(data_lines)

    # 计算切分点
    total_lines = len(data_lines)
    split_index = int(total_lines * 0.8)  # 80% 用于训练

    train_data = data_lines[:split_index]
    val_data = data_lines[split_index:]

    print(f"总数据量: {total_lines}")
    print(f"训练集 (80%): {len(train_data)} 条 -> {train_output}")
    print(f"验证集 (20%): {len(val_data)} 条 -> {val_output}")

    # 写入训练集
    with open(train_output, 'w', encoding='utf-8') as f:
        f.write(root_header)  # 写入表头
        f.writelines(train_data)

    # 写入验证集
    with open(val_output, 'w', encoding='utf-8') as f:
        f.write(root_header)  # 写入表头
        f.writelines(val_data)

    print("切分完成！")


if __name__ == '__main__':
    split_dataset()