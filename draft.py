# find /workspace/xcodec2n/LibriSpeech/train-clean-100 -name "*.flac" | while read -r file; do
#     relative_path=${file#/workspace/xcodec2n/LibriSpeech/}
#     echo "{\"audio_filepath\": \"$relative_path\"}" >> /workspace/xcodec2n/filelists/train_filelist.json
# done

# find /workspace/xcodec2n/LibriSpeech/test-clean -name "*.flac" | while read -r file; do
#     relative_path=${file#/workspace/xcodec2n/LibriSpeech/}
#     echo "{\"audio_filepath\": \"$relative_path\"}" >> /workspace/xcodec2n/filelists/test_filelist.json
# done

# # 统计 train_filelist.json 的行数
# wc -l /workspace/xcodec2n/filelists/train_filelist.json

# # 随机抽取 1000 条作为验证集
# shuf /workspace/xcodec2n/filelists/train_filelist.json | head -n 1000 > /workspace/xcodec2n/filelists/val_filelist.json

# # 备份原始训练集 filelist
# cp /workspace/xcodec2n/filelists/train_filelist.json /workspace/xcodec2n/filelists/train_filelist.json.bak

# # 移除验证集条目
# grep -Fxv -f /workspace/xcodec2n/filelists/val_filelist.json /workspace/xcodec2n/filelists/train_filelist.json.bak > /workspace/xcodec2n/filelists/train_filelist_new.json

# # 替换原始文件
# mv /workspace/xcodec2n/filelists/train_filelist_new.json /workspace/xcodec2n/filelists/train_filelist.json

# 改默认路径到workspace里
# export HF_HOME=/workspace/xcodec2n/models/huggingface
# export TRANSFORMERS_CACHE=/workspace/xcodec2n/models/huggingface
# export TORCH_HOME=/workspace/xcodec2n/models/torch
# export S3PRL_CACHE_DIR=/workspace/xcodec2n/models/s3prl

#查看是否成功
# env | egrep "HOME|HF|TORCH|S3PRL"

#生成val集
#shuf /workspace/xcodec2n/filelists/train_filelist.json | head -n 1000 > /workspace/xcodec2n/filelists/val_filelist.json

# mkdir -p /workspace/xcodec2n/models/huggingface
#添加到 /workspace/.bashrc
#echo "export HF_HOME=/workspace/xcodec2n/models/huggingface" >> /workspace/.bashrc
#source /workspace/.bashrc

# 设置 s3prl 缓存路径：
# export TORCH_HOME=/workspace/xcodec2n/models/torch
# mkdir -p /workspace/xcodec2n/models/torch

# 添加到 /workspace/.bashrc
# echo "export TORCH_HOME=/workspace/xcodec2n/models/torch" >> /workspace/.bashrc
# source /workspace/.bashrc

#-i https://pypi.tuna.tsinghua.edu.cn/simple


import hydra
from hydra import initialize, compose
from data_module import DataModule

# 初始化配置路径
with initialize(config_path="config", version_base="1.1"):
    cfg = compose(config_name="default")
    data_module = DataModule(cfg)
    train_loader = data_module.val_dataloader()
    for batch in train_loader:
        print(batch['wav'].shape, batch['feats'].shape)
        break