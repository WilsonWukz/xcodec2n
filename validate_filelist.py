#!/usr/bin/env python3
# 保存为 validate_filelist.py
import os, sys, json, argparse, traceback
import soundfile as sf
import torchaudio
from pathlib import Path
import re

def normalize_path_to_unix(p: str) -> str:
    if p is None:
        return p
    p = p.strip()
    if not p:
        return p
    if p.startswith('/') or p.startswith('file://'):
        return p
    m = re.match(r'^([A-Za-z]):[\\/](.*)$', p)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace('\\', '/')
        return f"/mnt/{drive}/{rest}"
    if '\\' in p:
        return p.replace('\\', '/')
    return p

def extract_paths(filelist_path):
    paths = []
    with open(filelist_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                wav = obj.get("audio_filepath", "") or obj.get("wav", "") or obj.get("path", "")
            except json.JSONDecodeError:
                wav = line.split('\t')[0]
            wav = normalize_path_to_unix(wav)
            paths.append(wav)
    return paths

def try_open(fullpath):
    """Try several open methods, return None if ok or exception string"""
    if not os.path.exists(fullpath):
        return f"NOT_EXIST"
    try:
        # 1) soundfile info (libsndfile)
        info = sf.info(fullpath)
    except Exception as e:
        return f"SOUNDFILE_ERR: {repr(e)}\n{traceback.format_exc(limit=1)}"
    try:
        # 2) basic read small chunk with soundfile (not full read to save time)
        with sf.SoundFile(fullpath) as f:
            # try to read first frame
            _ = f.read(frames=1, dtype='float32')
    except Exception as e:
        return f"SOUNDFILE_READ_ERR: {repr(e)}\n{traceback.format_exc(limit=1)}"
    try:
        # 3) torchaudio load (full)
        _wav, sr = torchaudio.load(fullpath)
    except Exception as e:
        return f"TORCHAUDIO_ERR: {repr(e)}\n{traceback.format_exc(limit=1)}"
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filelist", help="path to filelist (json lines or tsv)")
    parser.add_argument("--root", help="root dir to prepend if paths are relative", default=None)
    parser.add_argument("--max", type=int, default=None, help="max files to check")
    parser.add_argument("--out", default="bad_files.txt")
    args = parser.parse_args()

    paths = extract_paths(args.filelist)
    print(f"Total lines in filelist: {len(paths)}")
    bad = []
    checked = 0
    for p in paths:
        if args.max and checked >= args.max:
            break
        checked += 1
        if os.path.isabs(p):
            full = p
        else:
            if args.root:
                full = os.path.join(args.root, p)
            else:
                full = p
        full = os.path.normpath(full)
        res = try_open(full)
        if res is not None:
            bad.append((p, full, res))
            print(f"[BAD] {p} -> {full} : {res}")
        else:
            if checked % 200 == 0:
                print(f"Checked {checked} files... (last OK: {full})")
    print(f"Finished. Checked {checked} files. Bad files: {len(bad)}")
    if bad:
        with open(args.out, 'w', encoding='utf-8') as fw:
            for orig, full, err in bad:
                fw.write(f"{orig}\t{full}\t{err}\n")
        print(f"Problematic file list written to {args.out}")

if __name__ == "__main__":
    main()
