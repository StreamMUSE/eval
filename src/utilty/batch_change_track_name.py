#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

def find_generated_dirs(root: Path):
    for p in root.rglob('generated'):
        if p.is_dir():
            yield p

def run_rename(script: Path, src_dir: Path, old: str, new: str, out_dir: Path = None, backup: bool = False, dry=False):
    cmd = [sys.executable, str(script), str(src_dir), old, new, "--match", "exact"]
    if out_dir:
        cmd += ["--out-dir", str(out_dir)]
    if backup and not out_dir:
        cmd += ["--backup"]
    print("CMD:", " ".join(map(str, cmd)))
    if dry:
        return 0
    return subprocess.run(cmd).returncode

def main():
    parser = argparse.ArgumentParser(description="在每个 generated 文件夹上分两步替换 track 名称。")
    parser.add_argument("root", type=Path, help="根目录，递归查找名为 generated 的文件夹（例：realtime/baseline）")
    parser.add_argument("--change-script", type=Path, default=Path("src/utilty/change_track_name.py"), help="change_track_name.py 路径")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将要执行的命令，不真正执行")
    args = parser.parse_args()

    root = args.root
    script = args.change_script
    if not root.exists():
        print("根目录不存在：", root)
        raise SystemExit(2)
    if not script.exists():
        print("找不到 change_track_name 脚本：", script)
        raise SystemExit(2)

    gen_dirs = list(find_generated_dirs(root))
    if not gen_dirs:
        print("未找到任何 generated 文件夹。")
        return

    total = 0
    failed = 0
    for g in gen_dirs:
        total += 1
        print(f"\n处理：{g}")
        step1 = g.with_name(g.name + "_change_track_name")
        # 1) User Melody -> Piano, 输出到 step1（保留原始）
        rc = run_rename(script, g, "User Melody", "Piano", out_dir=step1, dry=args.dry_run)
        if rc != 0:
            print(f"第一步失败（{g}），返回码 {rc}")
            failed += 1
            continue
        # 2) 在 step1 内把 Mode Accompaniment -> Guitar，原地替换，带备份
        rc = run_rename(script, step1, "Mode Accompaniment", "Guitar", out_dir=None, backup=True, dry=args.dry_run)
        if rc != 0:
            print(f"第二步失败（{step1}），返回码 {rc}")
            failed += 1
            continue
        print(f"完成：{g} -> {step1}")

    print(f"\n全部完成。扫描 {total} 个 generated 文件夹，失败 {failed} 个。")

if __name__ == "__main__":
    main()