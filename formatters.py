"""出力フォーマッター（SRT/CSV/TXT/JSON）"""

import csv
import io
import json


def to_srt(segments):
    """話者ラベル付きSRT文字列を生成する。"""
    lines = []
    for i, seg in enumerate(segments, 1):
        start_ts = _format_srt_time(seg["start"])
        end_ts = _format_srt_time(seg["end"])
        text = f"[{seg['speaker']}] {seg['text']}"
        lines.append(f"{i}")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def to_csv(segments):
    """CSV文字列を生成する（話者,開始,終了,テキスト）。"""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["話者", "開始", "終了", "テキスト"])
    for seg in segments:
        writer.writerow([
            seg["speaker"],
            f"{seg['start']:.3f}",
            f"{seg['end']:.3f}",
            seg["text"],
        ])
    return buf.getvalue()


def to_txt(segments, show_timestamps=False):
    """プレーンテキストを生成する（話者ラベル付き）。"""
    lines = []
    current_speaker = None
    for seg in segments:
        if seg["speaker"] != current_speaker:
            current_speaker = seg["speaker"]
            lines.append(f"\n[{current_speaker}]")
        if show_timestamps:
            ts = _format_display_time(seg["start"])
            lines.append(f"[{ts}] {seg['text']}")
        else:
            lines.append(seg["text"])
    return "\n".join(lines).strip()


def to_json(segments):
    """詳細JSON文字列を生成する。"""
    return json.dumps(segments, ensure_ascii=False, indent=2)


def _format_display_time(seconds):
    """秒数を表示用タイムスタンプに変換する（HH:MM:SS）。"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_srt_time(seconds):
    """秒数をSRT形式のタイムスタンプに変換する（HH:MM:SS,mmm）。"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
