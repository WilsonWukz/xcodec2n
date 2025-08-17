#!/usr/bin/env python3
# check_filelist_readable.py
# Usage example:
# python check_filelist_readable.py \
#   --filelist /workspace/xcodec2n/filelists/train_filelist.json \
#   --root /workspace/xcodec2n/LibriSpeech \
#   --out bad_files.txt --max 1000

import os
import json
import argparse
import re
import soundfile as sf
import librosa
from pathlib import Path
import traceback

def _normalize_path_to_unix(p: str) -> str:
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

def extract_path_from_line(line: str):
    s = line.strip()
    if not s:
        return None
    # try JSON
    try:
        obj = json.loads(s)
        for k in ("audio_filepath", "audio", "wav", "path", "file"):
            if k in obj and obj[k]:
                return obj[k]
        # fallback: if JSON contains only one string value, try to find it
        # take first string value
        for v in obj.values():
            if isinstance(v, str) and v:
                return v
    except json.JSONDecodeError:
        pass
    # not JSON -> treat as TSV or plain path
    # split on tab first, then whitespace
    parts = re.split(r'\t', s, maxsplit=1)
    if parts and parts[0]:
        return parts[0]
    parts = re.split(r'\s+', s, maxsplit=1)
    if parts and parts[0]:
        return parts[0]
    return None

def try_read_one_frame(fullpath):
    """Try quick-read using soundfile, fallback to librosa if soundfile fails.
       Return (True, None) if ok; (False, error_str) if fail."""
    try:
        info = sf.info(fullpath)  # may raise
    except Exception as e:
        # capture traceback short
        return False, f"soundfile.info_err: {repr(e)}"
    try:
        with sf.SoundFile(fullpath) as f:
            _ = f.read(frames=1, dtype='float32')
        return True, None
    except Exception as e_sf:
        # fallback to librosa (may be slower)
        try:
            _ = librosa.load(fullpath, sr=None, mono=False, duration=0.01)
            return True, None
        except Exception as e_lb:
            return False, f"soundfile_read_err: {repr(e_sf)}; librosa_err: {repr(e_lb)}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--filelist', required=True, help='path to filelist (jsonl / tsv / plain)')
    p.add_argument('--root', default=None, help='root directory to prepend for relative paths')
    p.add_argument('--out', default='bad_files.txt', help='output file for problematic entries')
    p.add_argument('--max', type=int, default=None, help='max number of lines to check')
    p.add_argument('--skip_read', action='store_true', help='only check existence, skip reading attempt')
    p.add_argument('--verbose', action='store_true', help='print each OK line')
    args = p.parse_args()

    infile = args.filelist
    root = args.root
    outpath = args.out
    max_check = args.max
    skip_read = args.skip_read

    if not os.path.exists(infile):
        print("Filelist not found:", infile)
        return

    total = 0
    bad = []
    with open(infile, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if max_check and total >= max_check:
                break
            s = line.rstrip('\n')
            if not s.strip():
                continue
            path_field = extract_path_from_line(s)
            if path_field is None:
                bad.append((i, s, None, "NO_PATH_EXTRACTED"))
                if args.verbose:
                    print(f"[LINE {i}] NO_PATH_EXTRACTED")
                total += 1
                continue
            path_field = _normalize_path_to_unix(path_field)
            # build full path
            if os.path.isabs(path_field):
                full = path_field
            else:
                if root:
                    full = os.path.normpath(os.path.join(root, path_field))
                else:
                    full = os.path.normpath(path_field)
            status = "OK"
            err = None
            if not os.path.exists(full):
                status = "NOT_EXIST"
                err = None
            else:
                if not skip_read:
                    ok, errstr = try_read_one_frame(full)
                    if not ok:
                        status = "UNREADABLE"
                        err = errstr
            if status != "OK":
                bad.append((i, path_field, full, status if err is None else f"{status}:{err}"))
                print(f"[BAD {i}] {path_field} -> {full} : {status} {(' '+str(err)) if err else ''}")
            else:
                if args.verbose:
                    print(f"[OK {i}] {path_field}")
            total += 1
            if (i % 200) == 0:
                print(f"Checked {i} lines... bad so far: {len(bad)}")
    # write bad list
    with open(outpath, 'w', encoding='utf-8') as fo:
        for r in bad:
            # write: line_no \t original_path_field \t fullpath \t status
            fo.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\n")
    print(f"Done. Checked {total} lines. Bad entries: {len(bad)}")
    print(f"Wrote bad entries to: {outpath}")

if __name__ == "__main__":
    main()
