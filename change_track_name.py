# change track name in MIDI files
#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description="批量替换 MIDI 文件中某个 track 的名字")
    parser.add_argument("root", type=Path, help="包含 MIDI 文件的目录")
    parser.add_argument("old", help="要替换的原始 track 名称（大小写可选）")
    parser.add_argument("new", help="新的 track 名称")
    parser.add_argument("--glob", default="*.mid*", help="文件匹配模式，默认 *.mid*")
    parser.add_argument("--match", choices=("exact", "contains"), default="exact", help="匹配方式：exact 精确匹配（默认）或 contains 包含匹配")
    parser.add_argument("--out-dir", type=Path, default=None, help="若提供，则将修改后的文件写到该目录（保留原文件）")
    parser.add_argument("--backup", action="store_true", help="如果不指定 --out-dir，覆盖原文件前保留 .bak 备份")
    args = parser.parse_args()

    try:
        import pretty_midi
    except Exception:
        print("错误：需要安装 pretty_midi。运行 pip install pretty_midi", file=sys.stderr)
        sys.exit(2)

    if not args.root.is_dir():
        print("错误：指定路径不是目录。", file=sys.stderr)
        sys.exit(2)

    out_dir = None
    if args.out_dir:
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.root.glob(args.glob))
    if not files:
        print("未找到任何匹配的 MIDI 文件。")
        return

    old = args.old.strip().lower()
    total = 0
    changed = 0
    for fp in files:
        if not fp.is_file():
            continue
        total += 1
        try:
            midi = pretty_midi.PrettyMIDI(str(fp))
        except Exception as exc:
            print(f"跳过无法读取的文件 {fp.name}: {exc}", file=sys.stderr)
            continue

        modified = False
        for inst in midi.instruments:
            name = (inst.name or "").strip()
            if not name:
                continue
            if args.match == "exact":
                if name.lower() == old:
                    inst.name = args.new
                    modified = True
            else:  # contains
                if old in name.lower():
                    inst.name = name.replace(name, args.new) if name.lower() == old else name.replace(name, args.new)
                    # 更简单的替换：直接赋新名（也可以改为部分替换）
                    inst.name = args.new
                    modified = True

        if not modified:
            continue

        changed += 1
        if out_dir:
            dest = out_dir / fp.name
            midi.write(str(dest))
        else:
            if args.backup:
                bak = fp.with_suffix(fp.suffix + ".bak")
                try:
                    fp.rename(bak)
                except Exception:
                    # 无法重命名（权限等），先尝试复制备份
                    import shutil
                    shutil.copy2(fp, bak)
            midi.write(str(fp))

        print(f"Updated: {fp.name}")

    print(f"处理完成：共扫描 {total} 个文件，修改 {changed} 个文件。")

if __name__ == "__main__":
    main()