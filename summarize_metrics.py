#!/usr/bin/env python3
import os
import json
import csv
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import re

def get_nested(d, keys):
    cur = d
    for k in keys:
        if cur is None:
            return None
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur

def extract_metrics(summary):
    out = {}
    out['pitch_jsd'] = get_nested(summary, ['accompaniment_vs_groundtruth','pitch_jsd','mean'])
    out['onset_jsd'] = get_nested(summary, ['accompaniment_vs_groundtruth','onset_jsd','mean'])
    out['duration_jsd'] = get_nested(summary, ['accompaniment_vs_groundtruth','duration_jsd','mean'])
    out['prompt_groundtruth_txt_mean_distance'] = get_nested(summary, ['accompaniment_vs_groundtruth','prompt_groundtruth_continuation_polydis','txt_mean_distance', 'mean'])
    out['prompt_groundtruth_chd_mean_distance'] = get_nested(summary, ['accompaniment_vs_groundtruth','prompt_groundtruth_continuation_polydis','chd_mean_distance', 'mean'])
    out['prompt_generated_txt_mean_distance'] = get_nested(summary, ['accompaniment_vs_groundtruth','prompt_generated_continuation_polydis','txt_mean_distance', 'mean'])
    out['prompt_generated_chd_mean_distance'] = get_nested(summary, ['accompaniment_vs_groundtruth','prompt_generated_continuation_polydis','chd_mean_distance', 'mean'])
    fmd = get_nested(summary, ['accompaniment_vs_groundtruth','frechet_music_distance'])
    out['frechet_music_distance'] = fmd.get('mean') if isinstance(fmd, dict) else fmd
    out['rd'] = get_nested(summary, ['inter_track_continuity','auto_phrase_pairs','rd','mean'])
    out['vn'] = get_nested(summary, ['inter_track_continuity','auto_phrase_pairs','vn','mean'])
    out['consonant_ratio'] = get_nested(summary, ['melody_relationship','harmonicity','consonant_ratio','mean'])
    out['dissonant_ratio'] = get_nested(summary, ['melody_relationship','harmonicity','dissonant_ratio','mean'])
    out['unsupported_ratio'] = get_nested(summary, ['melody_relationship','harmonicity','unsupported_ratio','mean'])
    return out

def _collect_final_globals(root: Path) -> Dict[str, Dict[str, Optional[float]]]:
    """
    在 root 下查找所有 final-sys-results 目录中的 JSON 文件，
    返回一个 mapping: final_json_filename_stem -> { 'global_hit_rate':..., 'global_avg_backup':..., 'ISR_w':... }
    """
    mapping: Dict[str, Dict[str, Optional[float]]] = {}
    for final_dir in root.rglob("final-sys-results"):
        if not final_dir.is_dir():
            continue
        for jf in final_dir.glob("*.json"):
            try:
                obj = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            g = obj.get("global") or {}
            mapping[jf.stem] = {
                "global_hit_rate": g.get("global_hit_rate"),
                "global_avg_backup": g.get("global_avg_backup"),
                "ISR_w": g.get("ISR_w"),
                "source": str(jf),
            }
    return mapping

def _parse_interval_gen(name: str):
    """
    从文件名中解析 interval 与 gen_frame（返回 (interval:int|None, gen_frame:int|None)）。
    逻辑：
      - 先查 interval（如 interval1 或 interval_1）
      - 优先查找 interval 后面的 gen_frame_{n}
      - 若没有 gen_frame，查找 interval 后面的第一个 gen{n}（避免匹配到 prompt 的 gen_576）
      - 若仍无，退回全局第一次出现的 gen{n}
    """
    s = name.lower()
    # 找 interval
    m_int = re.search(r'interval[_-]?(\d+)', s)
    interval = int(m_int.group(1)) if m_int else None

    # 优先直接查 gen_frame_{n}
    m_gf = re.search(r'gen[_-]?frame[_-]?(\d+)', s)
    if m_gf:
        return interval, int(m_gf.group(1))

    # 如果找到了 interval，优先在 interval 之后查找第一个 gen{n}
    if m_int:
        start_pos = m_int.end()
        m_after = re.search(r'gen[_-]?(\d+)', s[start_pos:])
        if m_after:
            return interval, int(m_after.group(1))

    # 退回：全局查找第一个 gen{n}
    m_any = re.search(r'gen[_-]?(\d+)', s)
    if m_any:
        return interval, int(m_any.group(1))

    return None, None

def main(folder):
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"错误：{folder} 不是一个目录")

    files = [f for f in os.listdir(folder) if f.lower().endswith('.json') and os.path.isfile(os.path.join(folder, f))]
    if not files:
        print("未找到任何 JSON 文件，退出。")
        return

    metrics = [
        'pitch_jsd',
        'onset_jsd',
        'duration_jsd',
        'prompt_groundtruth_txt_mean_distance',
        'prompt_groundtruth_chd_mean_distance',
        'prompt_generated_txt_mean_distance',
        'prompt_generated_chd_mean_distance',
        'frechet_music_distance',
        'rd',
        'vn',
        'consonant_ratio',
        'dissonant_ratio',
        'unsupported_ratio'
    ]

    rows = []
    for fname in sorted(files):
        path = os.path.join(folder, fname)
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"警告：无法读取或解析 {fname}，跳过。错误：{e}")
            continue

        summary = data.get('summary')
        if summary is None:
            print(f"警告：{fname} 中没有 summary 键，跳过。")
            continue

        vals = extract_metrics(summary)
        row = {'filename': os.path.splitext(fname)[0]}
        for m in metrics:
            v = vals.get(m)
            row[m] = '' if v is None else v
        rows.append(row)

    if not rows:
        print("没有成功提取任何文件的数据，退出。")
        return

    # 收集 final-sys-results 下的 global metrics
    root_path = Path(folder)
    final_globals = _collect_final_globals(root_path)

    # 为每一行匹配 final json（文件名包含当前行的 filename），并添加三列：ISR (global_hit_rate), Staleness (global_avg_backup), ISR_w
    for r in rows:
        base = r['filename']
        matched = None

        # 首先尝试基于 interval / gen_frame 精确匹配
        base_interval, base_gen = _parse_interval_gen(base)
        if base_interval is not None and base_gen is not None:
            for final_stem, gdict in final_globals.items():
                f_interval, f_gen = _parse_interval_gen(final_stem)
                if f_interval is not None and f_gen is not None:
                    if (f_interval == base_interval) and (f_gen == base_gen):
                        matched = gdict
                        break

        # 若仍未匹配到，退回去用之前的包含/模糊匹配
        if matched is None:
            for final_stem, gdict in final_globals.items():
                if base.lower() in final_stem.lower() or final_stem.lower() in base.lower():
                    matched = gdict
                    break

        if matched is None:
            base_comp = base.replace('_', '').lower()
            for final_stem, gdict in final_globals.items():
                if base_comp in final_stem.replace('_', '').lower():
                    matched = gdict
                    break

        if matched:
            r['ISR'] = '' if matched.get("global_hit_rate") is None else matched.get("global_hit_rate")
            r['Staleness'] = '' if matched.get("global_avg_backup") is None else matched.get("global_avg_backup")
            r['ISR_w'] = '' if matched.get("ISR_w") is None else matched.get("ISR_w")
        else:
            print(f"警告：未找到与 {base} 匹配的 final-sys-results JSON 文件，相关列将留空。")
            r['ISR'] = ''
            r['Staleness'] = ''
            r['ISR_w'] = ''

    folder_basename = os.path.basename(folder.rstrip(os.sep))
    out_fname = f"{folder_basename} music quality summary.csv"
    out_path = os.path.join(folder, out_fname)

    header = ['filename'] + metrics + ['ISR', 'Staleness', 'ISR_w']
    with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"完成：已生成 CSV -> {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='汇总 folder 中 JSON 的 music quality summary 到 CSV（每行一个 JSON 文件，每列一个指标）')
    parser.add_argument('folder', help='要处理的文件夹路径')
    args = parser.parse_args()
    main(args.folder)