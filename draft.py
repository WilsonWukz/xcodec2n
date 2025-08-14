import os
import json
import soundfile as sf  # pip install soundfile

# 输入你的 wav 文件夹路径
wav_dir = "/mnt/e/dataset/manifest_wav"

# 输出 manifest 文件路径
output_manifest = "/mnt/e/dataset/manifests/manifest_debug.json"

# 逐个读取 wav 文件
with open(output_manifest, "w", encoding="utf-8") as f_out:
    for filename in os.listdir(wav_dir):
        if filename.lower().endswith(".wav"):
            wav_path = os.path.join(wav_dir, filename)

            # 获取时长
            try:
                sndf = sf.SoundFile(wav_path)
                duration = len(sndf) / sndf.samplerate
            except Exception as e:
                print(f"无法读取 {wav_path}: {e}")
                continue

            # 写一行 JSON
            entry = {
                "audio_filepath": wav_path,
                "text": "",  # 没有文本就留空
                "duration": round(duration, 3)
            }
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Manifest 已生成: {output_manifest}")
