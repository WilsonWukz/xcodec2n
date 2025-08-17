# tools/convert_windows_paths_to_wsl.py
import re
from pathlib import Path

in_txt = r"/mnt/e/dataset/manifests/librispeech_train_clean_100.txt"  # 原 txt（含 Windows 风格或混合）
out_txt = r"/mnt/e/dataset/manifests/librispeech_train_clean_100_unix.txt"

def convert_win_to_wsl(p: str) -> str:
    p = p.strip()
    if not p:
        return p
    # 如果已经看起来是 unix 风格，直接返回
    if p.startswith("/") or p.startswith("file://"):
        return p
    # Windows drive letter e.g. E:\path\to\file or E:/path/to/file
    m = re.match(r'^([A-Za-z]):[\\/](.*)$', p)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace('\\','/')
        return f"/mnt/{drive}/{rest}"
    # 如果不是 drive-letter，但是含反斜杠，替换为斜杠
    if '\\' in p:
        return p.replace('\\','/')
    return p

with open(in_txt, 'r', encoding='utf-8') as fi, open(out_txt, 'w', encoding='utf-8') as fo:
    for line in fi:
        orig = line.strip()
        if not orig:
            continue
        # 如果 filelist 包含其他字段用 tab 分隔，取第一个字段
        wav = orig.split('\t')[0]
        new = convert_win_to_wsl(wav)
        fo.write(new + '\n')

print("Wrote:", out_txt)
