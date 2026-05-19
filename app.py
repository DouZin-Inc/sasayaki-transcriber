"""Sasayaki — Gradio GUI エントリーポイント"""

import json
import locale
import os
import signal
import tempfile
import threading
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv, set_key

from formatters import to_csv, to_json, to_srt, to_txt
from transcriber import ALL_MODELS, DIARIZE_MODELS, _SUPPORTED_DIARIZE_MODELS, delete_model_cache, download_model, get_downloaded_models, get_gpu_list, is_whisperx_available, transcribe
from version import __version__

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(APP_DIR, ".env")
KEYWORDS_PATH = os.path.join(APP_DIR, "keywords.txt")
CRASH_LOG_PATH = os.path.join(APP_DIR, "crash.log")
HISTORY_DIR = os.path.join(APP_DIR, "history")
DEFAULT_OUTPUT_DIR = os.path.join(APP_DIR, "output")
DEFAULT_PROMPT = "処理結果に、句読点をいれてください。"
PROMPT_PATH = os.path.join(APP_DIR, "prompt.txt")
load_dotenv(ENV_PATH)


def _get_output_dir():
    """現在の出力先フォルダを返す。"""
    return os.environ.get("SASAYAKI_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR


def _set_output_dir(path):
    """出力先フォルダを .env に保存する。"""
    path = path.strip()
    if path:
        set_key(ENV_PATH, "SASAYAKI_OUTPUT_DIR", path)
        os.environ["SASAYAKI_OUTPUT_DIR"] = path


def _load_token():
    """保存済みトークンを読み込む。"""
    return os.getenv("HF_TOKEN", "")


def _save_token(token):
    """トークンを .env に保存する。"""
    if token:
        set_key(ENV_PATH, "HF_TOKEN", token)
        os.environ["HF_TOKEN"] = token


def _on_save_token(token):
    """トークン変更時に自動保存するハンドラ。"""
    if not token or not token.strip():
        return "未設定"
    _save_token(token.strip())
    return "✔ 保存済み"


def _on_save_output_dir(path):
    """出力先フォルダ変更時に自動保存するハンドラ。"""
    if not path or not path.strip():
        return "未設定（デフォルト使用）"
    path = path.strip()
    if not os.path.isabs(path):
        return "✗ 絶対パスで指定してください"
    _set_output_dir(path)
    os.makedirs(path, exist_ok=True)
    return "✔ 設定済み"


def _df_to_text(df):
    """Dataframe の値をテキストに変換する。"""
    if df is None:
        return ""
    if isinstance(df, str):
        return df
    # pandas DataFrame
    if hasattr(df, "values"):
        lines = [str(row[0]).strip() for row in df.values if str(row[0]).strip()]
        return "\n".join(lines)
    # list of lists
    lines = []
    for row in df:
        if isinstance(row, (list, tuple)) and row:
            val = str(row[0]).strip()
            if val:
                lines.append(val)
    return "\n".join(lines)


def _text_to_df(text):
    """テキストを Dataframe 形式（list of lists）に変換する。"""
    if not text or not text.strip():
        return []
    return [[kw.strip()] for kw in text.strip().splitlines() if kw.strip()]


def _load_keywords_as_df():
    """keywords.txt から Dataframe 形式で読み込む。"""
    return _text_to_df(_load_keywords())


def _on_save_keywords(df):
    """保存ボタンから呼ばれるハンドラ。"""
    _save_keywords(_df_to_text(df))
    gr.Info("キーワードを保存しました。")


def _on_select_keyword(df, evt: gr.SelectData):
    """Dataframe の行クリック時、入力欄に値をセットする。"""
    idx = evt.index[0]
    text = _df_to_text(df)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if 0 <= idx < len(lines):
        return lines[idx], idx
    return "", None


def _on_add_keyword(word, df, _sel_idx):
    """キーワードを1語追加するハンドラ。"""
    word = word.strip()
    if not word:
        gr.Info("キーワードを入力してください。")
        return "", df, None
    lines = [l.strip() for l in _df_to_text(df).splitlines() if l.strip()]
    if word in lines:
        gr.Info(f"「{word}」は既に登録されています。")
        return "", df, None
    lines.append(word)
    text = "\n".join(lines)
    _save_keywords(text)
    return "", _text_to_df(text), None


def _on_update_keyword(word, df, sel_idx):
    """選択中のキーワードを更新するハンドラ。"""
    word = word.strip()
    if sel_idx is None:
        gr.Info("更新するキーワードをクリックで選択してください。")
        return word, df, sel_idx
    if not word:
        gr.Info("キーワードを入力してください。")
        return word, df, sel_idx
    lines = [l.strip() for l in _df_to_text(df).splitlines() if l.strip()]
    if 0 <= sel_idx < len(lines):
        lines[sel_idx] = word
    text = "\n".join(lines)
    _save_keywords(text)
    gr.Info("キーワードを更新しました。")
    return "", _text_to_df(text), None


def _on_delete_keyword(df, sel_idx):
    """選択中のキーワードを削除するハンドラ。"""
    if sel_idx is None:
        gr.Info("削除するキーワードをクリックで選択してください。")
        return "", df, None
    lines = [l.strip() for l in _df_to_text(df).splitlines() if l.strip()]
    if 0 <= sel_idx < len(lines):
        removed = lines.pop(sel_idx)
        gr.Info(f"「{removed}」を削除しました。")
    text = "\n".join(lines)
    _save_keywords(text)
    return "", _text_to_df(text), None


def _on_sort_keywords(df, method):
    """ソートボタンのハンドラ。ソート結果を反映し保存する。"""
    text = _df_to_text(df)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        gr.Info("ソートするキーワードがありません。")
        return df
    if method == "kana":
        try:
            locale.setlocale(locale.LC_COLLATE, "ja_JP.UTF-8")
        except locale.Error:
            pass
        lines.sort(key=locale.strxfrm)
    else:
        lines.sort()
    sorted_text = "\n".join(lines)
    _save_keywords(sorted_text)
    gr.Info("キーワードをソートしました。")
    return _text_to_df(sorted_text)


def _on_export_keywords(df):
    """エクスポートボタンのハンドラ。現在のキーワードをファイルとして返す。"""
    text = _df_to_text(df)
    _save_keywords(text)
    if not text.strip():
        raise gr.Error("エクスポートするキーワードがありません。")
    return KEYWORDS_PATH


def _on_import_keywords(file):
    """インポートボタンのハンドラ。ファイルからキーワードを読み込む。"""
    if file is None:
        raise gr.Error("ファイルを選択してください。")
    with open(file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise gr.Error("ファイルが空です。")
    _save_keywords(text)
    gr.Info(f"キーワードをインポートしました（{len(text.splitlines())}語）。")
    return _text_to_df(text)


def _load_keywords():
    """keywords.txt からキーワードを読み込む。"""
    if os.path.exists(KEYWORDS_PATH):
        with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def _save_keywords(text):
    """キーワードを keywords.txt に保存する。"""
    with open(KEYWORDS_PATH, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def _save_history(filename, segments):
    """文字起こし結果を履歴として保存する。"""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    now = datetime.now()
    safe_name = os.path.splitext(filename)[0][:50]
    entry_name = f"{now.strftime('%Y%m%d_%H%M%S')}_{safe_name}.json"
    data = {
        "filename": filename,
        "date": now.strftime("%Y-%m-%d %H:%M:%S"),
        "segments": segments,
    }
    path = os.path.join(HISTORY_DIR, entry_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _list_history():
    """保存済み履歴の一覧を返す（新しい順）。"""
    if not os.path.isdir(HISTORY_DIR):
        return []
    entries = []
    for name in sorted(os.listdir(HISTORY_DIR), reverse=True):
        if not name.endswith(".json"):
            continue
        path = os.path.join(HISTORY_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            label = f"{data['date']}  {data['filename']}"
            entries.append((label, name))
        except (json.JSONDecodeError, KeyError):
            continue
    return entries


def _on_load_history(entry_name, show_timestamps):
    """履歴を読み込んでプレビューに反映する。"""
    if not entry_name:
        raise gr.Error("履歴を選択してください。")
    path = os.path.join(HISTORY_DIR, entry_name)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data["segments"]
    preview = to_txt(segments, show_timestamps=show_timestamps)
    speakers = sorted(set(seg["speaker"] for seg in segments))
    base_name = os.path.splitext(data["filename"])[0]
    gr.Info(f"履歴を読み込みました: {data['filename']}")
    return (preview, segments, base_name,
            gr.update(choices=speakers, value=speakers),
            _segments_to_table(segments), gr.update(choices=speakers))


def _on_delete_history(entry_name):
    """履歴を削除する。"""
    if not entry_name:
        raise gr.Error("削除する履歴を選択してください。")
    path = os.path.join(HISTORY_DIR, entry_name)
    if os.path.exists(path):
        os.remove(path)
    gr.Info("履歴を削除しました。")
    entries = _list_history()
    return gr.update(choices=entries, value=None)


def _on_reexport(segments_data, text, formats, base_name, speaker_filter):
    """編集済みセグメントをファイルとしてエクスポートする（話者フィルター対応）。"""
    if not text.strip():
        raise gr.Error("エクスポートするテキストがありません。")
    if not formats:
        raise gr.Error("出力形式を1つ以上選択してください。")

    # フォルダ選択ダイアログ（キャンセル時はデフォルト）
    out_dir = _pick_save_folder() or _get_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    name = base_name or "edited"

    # セグメントデータがあれば構造化形式も出力可能
    if segments_data:
        # 話者フィルターが設定されていれば適用
        if speaker_filter:
            filtered = [seg for seg in segments_data
                        if seg["speaker"] in speaker_filter]
            if not filtered:
                raise gr.Error("選択した話者のセグメントがありません。")
            segments_data = filtered
        _export_segments(segments_data, formats, name, out_dir)
        _open_folder(out_dir)
        gr.Info(f"{out_dir} に保存しました。")
        return []

    # セグメントなし → TXT のみ
    out_path = os.path.join(out_dir, f"{name}.txt")
    with open(out_path, "w", encoding="utf-8-sig") as f:
        f.write(text)
    _open_folder(out_dir)
    gr.Info(f"{out_dir} に保存しました。")
    return []


def _load_prompt():
    """保存済みプロンプトを読み込む。未保存ならデフォルト値を返す。"""
    if os.path.exists(PROMPT_PATH):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    return DEFAULT_PROMPT


def _save_prompt(text):
    """プロンプトを保存する。"""
    with open(PROMPT_PATH, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def _build_initial_prompt(prompt_text, keywords_text):
    """プロンプトとキーワードを結合して initial_prompt を構築する。"""
    parts = []
    if prompt_text.strip():
        parts.append(prompt_text.strip())
    keywords = [k.strip() for k in keywords_text.splitlines() if k.strip()]
    if keywords:
        parts.append("、".join(keywords))
    return " ".join(parts) if parts else None


def _load_crash_log():
    """crash.log の内容を読み込む。"""
    if os.path.exists(CRASH_LOG_PATH):
        with open(CRASH_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return "(ログファイルがありません)"


def _on_download_crash_log():
    """crash.log をダウンロード用に返す。"""
    if not os.path.exists(CRASH_LOG_PATH):
        raise gr.Error("ログファイルがありません。")
    return CRASH_LOG_PATH


def _build_diarize_choices():
    """話者分離モデルドロップダウンの選択肢を生成する。"""
    whisperx_installed = is_whisperx_available()
    choices = []
    for label, value in DIARIZE_MODELS:
        if value == "whisperx" and not whisperx_installed:
            choices.append((f"{label}（要インストール）", value))
        else:
            choices.append((label, value))
    return choices


def _on_diarize_model_change(diarize_model):
    """話者分離モデル変更時のハンドラ。WhisperX 未インストール時に案内を表示。"""
    if diarize_model == "whisperx" and not is_whisperx_available():
        gr.Warning(
            "WhisperX がインストールされていません。"
            "「設定」タブの「WhisperX インストール」ボタンから"
            "インストールできます。"
        )


def _on_install_whisperx(progress=gr.Progress()):
    """WhisperX をインストールするハンドラ。"""
    if is_whisperx_available():
        gr.Info("WhisperX は既にインストール済みです。")
        return (gr.update(choices=_build_diarize_choices()),
                "WhisperX: インストール済み")

    python_exe = os.path.join(APP_DIR, "python", "Scripts", "pip.exe")
    if not os.path.isfile(python_exe):
        python_exe = "pip"

    import subprocess as _sp
    progress(0.1, desc="WhisperX をインストール中... しばらくお待ちください")
    result = _sp.run(
        [python_exe, "install", "whisperx", "transformers"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        raise gr.Error(
            f"WhisperX のインストールに失敗しました:\n{result.stderr[-500:]}"
        )

    progress(1.0, desc="インストール完了")
    gr.Info("WhisperX のインストールが完了しました。アプリを再起動してください。")
    return (gr.update(choices=_build_diarize_choices()),
            "WhisperX: インストール済み")


def _build_download_choices():
    """ダウンロード用ドロップダウンの選択肢を生成する。"""
    downloaded = set(get_downloaded_models())
    choices = []
    for m in ALL_MODELS:
        if m in downloaded:
            choices.append((f"{m} (ダウンロード済み)", m))
        else:
            choices.append((f"{m} (未ダウンロード)", m))
    return choices


def _on_download_model(model_size, progress=gr.Progress()):
    """モデルダウンロードボタンのハンドラ。"""
    def on_progress(value, msg):
        progress(value, desc=msg)
    download_model(model_size, progress_callback=on_progress)
    gr.Info(f"モデル {model_size} のダウンロードが完了しました。")
    downloaded = get_downloaded_models()
    return (
        gr.update(choices=downloaded, value=model_size),
        gr.update(choices=_build_download_choices(), value=model_size),
    )


def _on_delete_cache(model_size):
    """モデルキャッシュ削除ボタンのハンドラ。"""
    delete_model_cache(model_size)
    gr.Info(f"モデル {model_size} のキャッシュを削除しました。")
    downloaded = get_downloaded_models()
    return (
        gr.update(choices=downloaded, value=downloaded[-1] if downloaded else None),
        gr.update(choices=_build_download_choices(), value=model_size),
    )


def _open_folder(path):
    """フォルダをエクスプローラーで開く。"""
    import subprocess
    import sys
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def _pick_save_folder():
    """pywebview のフォルダ選択ダイアログを開く。選択されなければ None を返す。"""
    try:
        import webview
        if webview.windows:
            default = _get_output_dir()
            os.makedirs(default, exist_ok=True)
            result = webview.windows[0].create_file_dialog(
                webview.FOLDER_DIALOG,
                directory=default,
            )
            if result and len(result) > 0:
                return result[0]
    except Exception:
        pass
    return None


def _export_segments(segments, formats, base_name, out_dir=None):
    """セグメントデータをフォーマットしてファイルに書き出す。"""
    if out_dir is None:
        out_dir = _get_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    output_files = []
    format_map = {
        "SRT": ("srt", to_srt),
        "CSV": ("csv", to_csv),
        "TXT": ("txt", to_txt),
        "JSON": ("json", to_json),
    }
    for fmt in formats:
        ext, formatter = format_map[fmt]
        content = formatter(segments)
        out_path = os.path.join(out_dir, f"{base_name}.{ext}")
        with open(out_path, "w", encoding="utf-8-sig") as f:
            f.write(content)
        output_files.append(out_path)
    return output_files


def run(files, mic_audio, model_size, num_speakers, hf_token,
        prompt_text, keywords, language, gpu_device, show_timestamps,
        enable_diarization, diarize_model, word_timestamps,
        progress=gr.Progress()):
    """文字起こしを実行する。エクスポートは別途エクスポートボタンから行う。"""

    has_files = files is not None and (not isinstance(files, list) or len(files) > 0)
    has_mic = mic_audio is not None

    if not has_files and not has_mic:
        raise gr.Error("ファイルを選択するか、マイクで録音してください。")

    # マイク録音の場合はファイルリストに変換
    if has_mic and not has_files:
        files = [mic_audio]

    # トークン保存
    _save_token(hf_token)

    if enable_diarization and diarize_model not in _SUPPORTED_DIARIZE_MODELS:
        raise gr.Error(
            f"選択された話者分離モデルは現在未対応です（対応予定）。"
        )
    if enable_diarization and diarize_model == "whisperx" and not is_whisperx_available():
        raise gr.Error(
            "WhisperX がインストールされていません。"
            "「設定」タブの「WhisperX をインストール」ボタンから"
            "インストールしてください。"
        )
    if enable_diarization and not hf_token:
        raise gr.Error(
            "話者分離には HuggingFace トークンが必要です。設定欄に入力してください。"
        )

    # プロンプト・キーワード保存 & 構築
    _save_prompt(prompt_text)
    keywords_text = _df_to_text(keywords)
    _save_keywords(keywords_text)
    initial_prompt = _build_initial_prompt(prompt_text, keywords_text)

    # 単一ファイルの場合もリスト化
    if not isinstance(files, list):
        files = [files]

    all_previews = []
    all_segments = []

    try:
        for idx, file in enumerate(files):
            file_label = f"[{idx + 1}/{len(files)}] {os.path.basename(file)}"

            # 進捗コールバック（ファイル番号付き）
            def on_progress(value, msg, _label=file_label):
                progress(value, desc=f"{_label}: {msg}")

            # 文字起こし実行
            segments = transcribe(
                audio_path=file,
                model_size=model_size,
                num_speakers=num_speakers,
                hf_token=hf_token,
                initial_prompt=initial_prompt,
                gpu_device=gpu_device,
                language=language,
                progress_callback=on_progress,
                enable_diarization=enable_diarization,
                diarize_model=diarize_model,
                word_timestamps=word_timestamps,
            )

            # 結果を収集
            base_name = os.path.splitext(os.path.basename(file))[0]
            all_segments.extend(segments)
            all_previews.append(f"=== {os.path.basename(file)} ===\n{to_txt(segments, show_timestamps=show_timestamps)}")

            # ファイルごとに履歴保存
            _save_history(os.path.basename(file), segments)
    except gr.Error:
        raise
    except Exception as e:
        import traceback
        error_text = (
            f"エラーが発生しました（プレビュー欄からコピーできます）:\n\n"
            f"{type(e).__name__}: {e}\n\n"
            f"{traceback.format_exc()}"
        )
        return ([], error_text, "", None, "",
                gr.update(), gr.update(), [], gr.update())

    preview = "\n\n".join(all_previews)

    # 使用モデル情報を構築
    _diarize_labels = {
        "pyannote/speaker-diarization-3.1": "pyannote 3.1",
        "whisperx": "WhisperX",
    }
    diarize_label = _diarize_labels.get(diarize_model, diarize_model) if enable_diarization else "なし"
    model_info = f"{model_size} / {diarize_label}"

    # 話者一覧を抽出
    speakers = sorted(set(seg["speaker"] for seg in all_segments))
    last_base_name = os.path.splitext(os.path.basename(files[-1]))[0]

    history_choices = _list_history()

    return ([], preview, model_info, all_segments, last_base_name,
            gr.update(choices=speakers, value=speakers),
            gr.update(choices=history_choices, value=None),
            _segments_to_table(all_segments),
            gr.update(choices=speakers))


def _on_replace(segments_data, speaker_replace, text_replace,
                 show_timestamps):
    """話者名・テキストを一括置換する。"""
    if not segments_data:
        raise gr.Error("先に文字起こしを実行してください。")

    # 置換マップを構築（"OLD=NEW" 形式、1行1組）
    def _parse_map(text):
        result = {}
        for line in text.strip().splitlines():
            if "=" in line:
                old, new = line.split("=", 1)
                if old.strip():
                    result[old.strip()] = new.strip()
        return result

    speaker_map = _parse_map(speaker_replace)
    text_map = _parse_map(text_replace)

    if not speaker_map and not text_map:
        raise gr.Error("置換する内容を入力してください。")

    # セグメントの話者名・テキストを置換
    new_segments = []
    for seg in segments_data:
        new_seg = dict(seg)
        if new_seg["speaker"] in speaker_map:
            new_seg["speaker"] = speaker_map[new_seg["speaker"]]
        for old_text, new_text in text_map.items():
            new_seg["text"] = new_seg["text"].replace(old_text, new_text)
        new_segments.append(new_seg)

    count_speaker = sum(1 for s in speaker_map if s)
    count_text = sum(1 for t in text_map if t)
    speakers = sorted(set(seg["speaker"] for seg in new_segments))
    preview = to_txt(new_segments, show_timestamps=show_timestamps)
    gr.Info(f"置換完了（話者名: {count_speaker}件、テキスト: {count_text}件）")
    return (preview, new_segments,
            gr.update(choices=speakers, value=speakers),
            _segments_to_table(new_segments), gr.update(choices=speakers))



def _on_speaker_filter_change(segments_data, speaker_filter, show_timestamps):
    """話者フィルター変更時にプレビューを更新する。"""
    if not segments_data:
        return ""
    if speaker_filter:
        filtered = [seg for seg in segments_data
                    if seg["speaker"] in speaker_filter]
    else:
        filtered = segments_data
    return to_txt(filtered, show_timestamps=show_timestamps)


def _insert_equals(text):
    """テキスト末尾に半角 = を挿入する。"""
    return text + "="


def _format_time(seconds):
    """秒数を HH:MM:SS.mmm 形式に変換する。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _parse_time(time_str):
    """HH:MM:SS.mmm または秒数を float に変換する。"""
    time_str = time_str.strip()
    if ":" in time_str:
        parts = time_str.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    return float(time_str)


def _segments_to_table(segments):
    """セグメントリストをテーブル表示用データに変換する。"""
    if not segments:
        return []
    return [
        [i + 1, seg["speaker"], _format_time(seg["start"]),
         _format_time(seg["end"]), seg["text"]]
        for i, seg in enumerate(segments)
    ]


def _on_select_segment(table_data, evt: gr.SelectData):
    """セグメントテーブルの行クリック時のハンドラ。"""
    idx = evt.index[0]
    if table_data is None or idx >= len(table_data):
        return None, "", "", gr.update(label="分割文字位置")
    row = table_data.values.tolist()[idx] if hasattr(table_data, "values") else table_data[idx]
    text = str(row[4])
    char_count = len(text)
    return idx, str(row[1]), text, gr.update(label=f"分割文字位置（全{char_count}文字）")


def _on_update_segment(segments_data, selected_idx, new_speaker, new_text,
                       show_timestamps):
    """選択したセグメントの話者・テキストを更新する。"""
    if selected_idx is None:
        raise gr.Error("セグメントを選択してください。")
    idx = int(selected_idx)
    if not segments_data or idx >= len(segments_data):
        raise gr.Error("無効なセグメントです。")

    new_segments = [dict(seg) for seg in segments_data]
    new_segments[idx]["speaker"] = new_speaker
    new_segments[idx]["text"] = new_text

    table = _segments_to_table(new_segments)
    preview = to_txt(new_segments, show_timestamps=show_timestamps)
    speakers = sorted(set(seg["speaker"] for seg in new_segments))
    gr.Info(f"セグメント #{idx + 1} を更新しました。")
    return (new_segments, table, preview,
            gr.update(choices=speakers, value=speakers))


def _on_merge_segment(segments_data, selected_idx, show_timestamps):
    """選択したセグメントを次のセグメントと統合する。"""
    if selected_idx is None:
        raise gr.Error("統合するセグメントを選択してください。")
    idx = int(selected_idx)
    if not segments_data or idx >= len(segments_data) - 1:
        raise gr.Error("最後のセグメントは統合できません。")

    new_segments = [dict(seg) for seg in segments_data]
    seg1 = new_segments[idx]
    seg2 = new_segments[idx + 1]
    merged = {
        "speaker": seg1["speaker"],
        "start": seg1["start"],
        "end": seg2["end"],
        "text": seg1["text"] + seg2["text"],
    }
    new_segments = new_segments[:idx] + [merged] + new_segments[idx + 2:]

    table = _segments_to_table(new_segments)
    preview = to_txt(new_segments, show_timestamps=show_timestamps)
    speakers = sorted(set(seg["speaker"] for seg in new_segments))
    gr.Info(f"セグメント #{idx + 1} と #{idx + 2} を統合しました。")
    return (new_segments, table, preview, None,
            gr.update(choices=speakers, value=speakers))


def _on_split_segment(segments_data, selected_idx, split_time_str,
                      split_char_pos, show_timestamps):
    """選択したセグメントを指定時刻または文字位置で分割する。"""
    if selected_idx is None:
        raise gr.Error("分割するセグメントを選択してください。")
    idx = int(selected_idx)
    if not segments_data or idx >= len(segments_data):
        raise gr.Error("無効なセグメントです。")

    has_time = split_time_str and split_time_str.strip()
    has_pos = split_char_pos is not None and split_char_pos > 0

    if not has_time and not has_pos:
        raise gr.Error("分割時刻または分割文字位置を入力してください。")

    seg = segments_data[idx]
    text = seg["text"]
    duration = seg["end"] - seg["start"]

    if has_time:
        # 時刻指定: 従来通り
        try:
            split_time = _parse_time(split_time_str)
        except (ValueError, IndexError):
            raise gr.Error("分割時刻の形式が正しくありません。")

        if split_time <= seg["start"] or split_time >= seg["end"]:
            raise gr.Error(
                f"分割時刻はセグメントの範囲内 "
                f"({_format_time(seg['start'])} ~ {_format_time(seg['end'])}) "
                f"で指定してください。"
            )

        ratio = (split_time - seg["start"]) / duration
        split_pos = max(1, int(len(text) * ratio))
    else:
        # 文字位置指定: 時刻を按分で算出
        split_pos = int(split_char_pos)
        if split_pos < 1 or split_pos >= len(text):
            raise gr.Error(
                f"分割文字位置は 1〜{len(text) - 1} の範囲で"
                f"指定してください（全{len(text)}文字）。"
            )
        ratio = split_pos / len(text)
        split_time = seg["start"] + duration * ratio

    seg1 = {"speaker": seg["speaker"], "start": seg["start"],
            "end": split_time, "text": text[:split_pos]}
    seg2 = {"speaker": seg["speaker"], "start": split_time,
            "end": seg["end"], "text": text[split_pos:]}

    new_segments = [dict(s) for s in segments_data]
    new_segments = new_segments[:idx] + [seg1, seg2] + new_segments[idx + 1:]

    table = _segments_to_table(new_segments)
    preview = to_txt(new_segments, show_timestamps=show_timestamps)
    speakers = sorted(set(seg["speaker"] for seg in new_segments))
    gr.Info(f"セグメント #{idx + 1} を分割しました。")
    return (new_segments, table, preview, None,
            gr.update(choices=speakers, value=speakers))


def _on_delete_segment(segments_data, selected_idx, show_timestamps):
    """選択したセグメントを削除する。"""
    if selected_idx is None:
        raise gr.Error("削除するセグメントを選択してください。")
    idx = int(selected_idx)
    if not segments_data or idx >= len(segments_data):
        raise gr.Error("無効なセグメントです。")

    new_segments = [dict(s) for s in segments_data]
    new_segments.pop(idx)

    table = _segments_to_table(new_segments)
    preview = to_txt(new_segments, show_timestamps=show_timestamps) if new_segments else ""
    speakers = sorted(set(seg["speaker"] for seg in new_segments)) if new_segments else []
    gr.Info(f"セグメント #{idx + 1} を削除しました。")
    return (new_segments, table, preview, None,
            gr.update(choices=speakers, value=speakers))


def _on_toggle_diarization(enabled):
    """話者分離トグルの切替ハンドラ。"""
    return (
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
    )


def _on_toggle_timestamp(segments_data, show_timestamps):
    """タイムスタンプ表示の切替ハンドラ。"""
    if not segments_data:
        return ""
    return to_txt(segments_data, show_timestamps=show_timestamps)


def build_ui():
    """Gradio UIを構築する。"""
    downloaded = get_downloaded_models(verify=True)

    with gr.Blocks(
        title="Sasayaki — 文字起こし＆話者分離",
        analytics_enabled=False,
    ) as app:
        gr.Markdown(f"# Sasayaki（ささやき） v{__version__}")

        segments_state = gr.State(value=None)
        base_name_state = gr.State(value="")

        with gr.Row():
            # ==========================================
            # 左カラム: タブ（文字起こし / 後処理 / 設定 / ヘルプ）
            # ==========================================
            with gr.Column(scale=1):
                with gr.Tabs():
                    # ==========================================
                    # メインタブ: 文字起こし
                    # ==========================================
                    with gr.Tab("文字起こし"):
                        with gr.Tabs():
                            with gr.Tab("ファイル"):
                                file_input = gr.File(
                                    label="動画/音声ファイル（複数選択可）",
                                    file_count="multiple",
                                    file_types=[
                                        ".mp4", ".mkv", ".avi", ".mov", ".webm",
                                        ".mp3", ".wav", ".flac", ".m4a", ".ogg",
                                    ],
                                )
                            with gr.Tab("マイク録音"):
                                mic_input = gr.Audio(
                                    sources=["microphone"],
                                    type="filepath",
                                    label="録音（停止ボタンで録音完了）",
                                )

                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                choices=downloaded,
                                value=downloaded[-1] if downloaded else None,
                                label="モデル",
                                scale=2,
                            )
                            language_dropdown = gr.Dropdown(
                                choices=[
                                    ("日本語", "ja"),
                                    ("英語", "en"),
                                    ("中国語", "zh"),
                                    ("韓国語", "ko"),
                                    ("自動検出", ""),
                                ],
                                value="ja",
                                label="言語",
                                scale=1,
                            )

                        with gr.Row():
                            diarization_toggle = gr.Checkbox(
                                label="話者分離を有効にする",
                                value=False,
                                scale=1,
                            )
                            diarize_model_dropdown = gr.Dropdown(
                                choices=_build_diarize_choices(),
                                value=DIARIZE_MODELS[0][1],
                                label="話者分離モデル",
                                interactive=False,
                                scale=2,
                            )

                        with gr.Row():
                            num_speakers_input = gr.Number(
                                value=0,
                                label="話者数（0で自動推定）",
                                precision=0,
                                minimum=0,
                                maximum=20,
                                interactive=False,
                                scale=1,
                            )
                            word_timestamps_toggle = gr.Checkbox(
                                label="高精度タイムスタンプ",
                                value=False,
                                info="有効にすると字幕のタイミング精度が向上しますが、処理時間が増加します",
                                scale=1,
                            )

                        with gr.Row():
                            run_btn = gr.Button("文字起こし開始", variant="primary", size="lg")
                            cancel_btn = gr.Button("キャンセル", variant="stop", size="lg")

                    # ==========================================
                    # 後処理タブ: 置換・履歴
                    # ==========================================
                    with gr.Tab("後処理"):
                        gr.Markdown("### セグメント編集")
                        gr.Markdown(
                            "行をクリックして選択し、話者やテキストを編集できます。"
                        )
                        segment_selected_idx = gr.State(value=None)
                        segments_table = gr.Dataframe(
                            headers=["#", "話者", "開始", "終了", "テキスト"],
                            datatype=["number", "str", "str", "str", "str"],
                            col_count=(5, "fixed"),
                            interactive=False,
                            label="セグメント一覧",
                            column_widths=["50px", "100px", "110px", "110px", None],
                            wrap=True,
                        )
                        with gr.Row():
                            seg_speaker_input = gr.Dropdown(
                                choices=[],
                                allow_custom_value=True,
                                label="話者",
                                scale=1,
                            )
                            seg_text_input = gr.Textbox(
                                label="テキスト",
                                lines=1,
                                scale=3,
                            )
                        with gr.Row():
                            seg_update_btn = gr.Button("更新", size="sm")
                            seg_merge_btn = gr.Button("次と統合", size="sm")
                            seg_split_time = gr.Textbox(
                                label="分割時刻",
                                placeholder="例: 00:01:30.000",
                                lines=1,
                                scale=1,
                            )
                            seg_split_pos = gr.Number(
                                label="分割文字位置",
                                precision=0,
                                minimum=1,
                                placeholder="何文字目で分割",
                                scale=1,
                            )
                            seg_split_btn = gr.Button("分割", size="sm")
                            seg_delete_btn = gr.Button(
                                "削除", size="sm", variant="stop",
                            )

                        speaker_filter = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label="話者フィルター",
                            info="チェックを外すとプレビュー・エクスポートから除外されます",
                        )

                        with gr.Accordion("一括置換", open=False):
                            speaker_replace_input = gr.Textbox(
                                label="話者名の置換（1行1組、= で区切る）",
                                placeholder="SPEAKER_00=山田\nSPEAKER_01=田中",
                                lines=3,
                            )
                            speaker_eq_btn = gr.Button(
                                "= を挿入", size="sm", min_width=80,
                            )
                            text_replace_input = gr.Textbox(
                                label="テキストの一括置換（1行1組、= で区切る）",
                                placeholder="えーと=（削除）\nわたくし=私",
                                lines=3,
                            )
                            text_eq_btn = gr.Button(
                                "= を挿入", size="sm", min_width=80,
                            )
                            replace_btn = gr.Button("一括置換", variant="primary")

                        with gr.Accordion("履歴", open=False):
                            history_dropdown = gr.Dropdown(
                                choices=_list_history(),
                                label="保存された履歴",
                                info="文字起こし完了時に自動保存されます",
                            )
                            with gr.Row():
                                history_load_btn = gr.Button("読み込み")
                                history_delete_btn = gr.Button("削除", variant="stop")

                    # ==========================================
                    # 辞書タブ: キーワード辞書管理
                    # ==========================================
                    with gr.Tab("辞書"):
                        gr.Markdown("### キーワード辞書")
                        gr.Markdown("人名・固有名詞を登録すると認識精度が向上します。行をクリックして選択できます。")
                        selected_keyword_idx = gr.State(value=None)
                        with gr.Row():
                            keyword_edit_input = gr.Textbox(
                                label="キーワード",
                                placeholder="例: 山田太郎",
                                lines=1,
                                scale=3,
                            )
                            keyword_add_btn = gr.Button("追加", size="sm", scale=0, min_width=70)
                            keyword_update_btn = gr.Button("更新", size="sm", scale=0, min_width=70)
                            keyword_delete_btn = gr.Button("削除", size="sm", variant="stop", scale=0, min_width=70)
                        keywords_input = gr.Dataframe(
                            value=_load_keywords_as_df(),
                            headers=["キーワード"],
                            datatype=["str"],
                            col_count=(1, "fixed"),
                            interactive=False,
                            label="登録済みキーワード",
                        )
                        with gr.Row():
                            save_keywords_btn = gr.Button("保存", size="sm")
                            sort_kana_btn = gr.Button("50音順ソート", size="sm")
                            sort_az_btn = gr.Button("A-Zソート", size="sm")
                        with gr.Accordion("インポート / エクスポート", open=False):
                            with gr.Row():
                                import_keywords_file = gr.File(
                                    label="インポート（.txt）",
                                    file_types=[".txt"],
                                    scale=1,
                                )
                                export_keywords_btn = gr.Button("エクスポート", size="sm", scale=0, min_width=120)
                            export_keywords_output = gr.File(
                                label="エクスポートファイル",
                                interactive=False,
                            )

                    # ==========================================
                    # 設定タブ: 初回設定・詳細
                    # ==========================================
                    with gr.Tab("設定"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### HuggingFace トークン")
                                _saved_token = _load_token()
                                hf_token_input = gr.Textbox(
                                    value=_saved_token,
                                    label="HuggingFace トークン",
                                    type="password",
                                    info="pyannote の話者分離に必要です。入力すると自動保存されます。",
                                )
                                token_status = gr.Textbox(
                                    value="✔ 保存済み" if _saved_token else "未設定",
                                    label="ステータス",
                                    interactive=False,
                                    max_lines=1,
                                )

                                gr.Markdown("### 出力先フォルダ")
                                output_dir_input = gr.Textbox(
                                    value=_get_output_dir(),
                                    label="エクスポート先",
                                    info="変更すると自動保存されます。",
                                )
                                output_dir_status = gr.Textbox(
                                    value="✔ 設定済み",
                                    label="ステータス",
                                    interactive=False,
                                    max_lines=1,
                                )

                                gr.Markdown("### GPU 設定")
                                gpu_choices = get_gpu_list()
                                default_gpu = gpu_choices[0][1] if len(gpu_choices) <= 1 else gpu_choices[0][1]
                                gpu_dropdown = gr.Dropdown(
                                    choices=gpu_choices,
                                    value=default_gpu,
                                    label="使用するデバイス",
                                    info="複数GPUがある場合はここで切り替えできます",
                                )

                                gr.Markdown("### モデルダウンロード")
                                download_model_dropdown = gr.Dropdown(
                                    choices=_build_download_choices(),
                                    value="large-v3",
                                    label="ダウンロードするモデル",
                                    info="大きいモデルほど高精度ですが、VRAMを多く使います",
                                )
                                with gr.Row():
                                    download_btn = gr.Button("ダウンロード開始", size="sm")
                                    delete_cache_btn = gr.Button("キャッシュ削除", size="sm", variant="stop")

                                gr.Markdown("### オプション パッケージ")
                                _whisperx_status = "インストール済み" if is_whisperx_available() else "未インストール"
                                whisperx_status_text = gr.Textbox(
                                    value=f"WhisperX: {_whisperx_status}",
                                    label="WhisperX（話者分離の一括実行）",
                                    info="文字起こし・アライメント・話者分離を一括で実行します",
                                    interactive=False,
                                )
                                whisperx_install_btn = gr.Button(
                                    "インストール",
                                    size="sm",
                                    interactive=not is_whisperx_available(),
                                )

                            with gr.Column(scale=1):
                                gr.Markdown("### プロンプト")
                                prompt_input = gr.Textbox(
                                    value=_load_prompt(),
                                    label="プロンプト（Whisper に渡す指示文）",
                                    lines=2,
                                    info="不要なら空欄にしてください。編集内容は自動保存されます。",
                                )

                                gr.Markdown("### ログ")
                                crash_log_text = gr.Textbox(
                                    value=_load_crash_log(),
                                    label="crash.log",
                                    lines=8,
                                    max_lines=20,
                                    interactive=False,
                                )
                                with gr.Row():
                                    log_reload_btn = gr.Button("再読み込み", size="sm")
                                    log_download_btn = gr.Button("ダウンロード", size="sm")
                                log_download_output = gr.File(
                                    visible=False,
                                )

                    # ==========================================
                    # ヘルプタブ
                    # ==========================================
                    with gr.Tab("ヘルプ"):
                        gr.Markdown("""\
### 基本操作

1. 「文字起こし」タブでファイルをドラッグ＆ドロップ、またはマイクで録音
2. モデル・言語・話者数・出力形式を選択
3. 「文字起こし開始」をクリック
4. 完了後、ファイルをダウンロード（結果は自動で履歴に保存されます）
5. 必要に応じて「後処理」タブで編集・置換

対応形式: `.mp4` `.mkv` `.avi` `.mov` `.webm` `.mp3` `.wav` `.flac` `.m4a` `.ogg`

---

### 話者分離モード

| モード | 特徴 |
|--------|------|
| **pyannote 3.1** | 標準の話者分離。faster-whisper で文字起こし後に pyannote で話者を識別 |
| **WhisperX** | 文字起こし・アライメント・話者分離を一括実行。初回は「設定」タブからインストールが必要 |

---

### 後処理機能

- **セグメント編集**: テーブルの行をクリックして話者名・テキストを個別編集
- **マージ**: 選択したセグメントと次のセグメントを統合
- **分割**: 指定した時刻でセグメントを2つに分割
- **削除**: 不要なセグメントを削除
- **一括置換**: `旧名=新名` 形式で話者名・テキストをまとめて置換
- **話者フィルター**: 特定の話者だけを選んでエクスポート

---

### 辞書機能

「辞書」タブで人名・固有名詞を登録すると、Whisper の認識精度が向上します。

- キーワードは1行1語で登録
- インポート/エクスポート（.txt）に対応
- 50音順・A-Z順でソート可能

---

### 初回セットアップ

1. 「設定」タブで HuggingFace トークンを入力
2. 「設定」タブでモデルをダウンロード
3. 準備完了 — 「文字起こし」タブで利用開始

---

### モデル比較

| モデル | VRAM目安 | 精度 | 速度 | おすすめ |
|--------|---------|------|------|---------|
| tiny | 1GB | ★★ | 最速 | テスト・確認用 |
| base | 1GB | ★★★ | 速い | GPU なしの PC |
| small | 2GB | ★★★★ | 普通 | バランス重視 |
| medium | 5GB | ★★★★★ | やや遅い | 高精度 |
| large-v3 | 10GB | ★★★★★+ | 遅い | 最高精度（GPU推奨） |

> **GPU がない場合**: `tiny` か `base` を選んでください。
>
> **NVIDIA GPU がある場合**: VRAM 容量に合ったモデルを選べます。タスクマネージャーで VRAM を確認できます。
>
> **高精度タイムスタンプ**: 有効にすると単語単位でセグメント境界を補正し、字幕のタイミング精度が向上します。

---

### HuggingFace トークンの取得手順

話者分離機能を使うには、初回のみ以下の設定が必要です（所要時間: 約5分）。

#### Step 1: HuggingFace アカウントを作成する

1. [HuggingFace 登録ページ](https://huggingface.co/join) にアクセス
2. メールアドレスとパスワードを入力してアカウントを作成
3. 届いた確認メールのリンクをクリックして認証を完了

#### Step 2: モデルの利用規約に同意する（3つとも必要）

1. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) を開く → 「**Agree and access repository**」をクリック
2. [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) を開く → 同様にクリック
3. [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) を開く → 同様にクリック

> **注意**: 3つとも同意が必要です。1つでも未同意だとエラーになります。

#### Step 3: アクセストークンを作成する

1. [トークン設定ページ](https://huggingface.co/settings/tokens) にアクセス
2. 「**Create new token**」→ Token type: 「**Read**」→ 「**Create token**」
3. 表示されたトークン（`hf_` で始まる文字列）をコピー

#### Step 4: Sasayaki に入力

「設定」タブの「HuggingFace トークン」欄に貼り付けて「トークンを保存」をクリック。

#### トラブルシューティング

| 症状 | 原因と対処 |
|------|-----------|
| `401 Unauthorized` | トークンが間違っているか期限切れ → 再作成 |
| `403 Forbidden` | モデルの利用規約に未同意 → Step 2 を確認 |
| `ConnectionError` | インターネット接続を確認 |
""")

            # ==========================================
            # 右カラム: ダウンロード・プレビュー（常時表示）
            # ==========================================
            with gr.Column(scale=1):
                output_files = gr.File(
                    label="ダウンロード",
                    file_count="multiple",
                    interactive=False,
                )
                model_info_text = gr.Textbox(
                    label="使用モデル",
                    interactive=False,
                    lines=1,
                    max_lines=1,
                )
                preview_text = gr.Textbox(
                    label="プレビュー（編集可能）",
                    lines=20,
                    max_lines=40,
                    interactive=True,
                )
                with gr.Row():
                    format_checkbox = gr.CheckboxGroup(
                        choices=["SRT", "CSV", "TXT", "JSON"],
                        value=["SRT"],
                        label="出力形式",
                    )
                with gr.Row():
                    reexport_btn = gr.Button("エクスポート", variant="primary", size="sm")
                    timestamp_toggle = gr.Checkbox(
                        label="タイムスタンプ表示",
                        value=False,
                    )

        diarize_model_dropdown.change(
            fn=_on_diarize_model_change,
            inputs=[diarize_model_dropdown],
        )
        whisperx_install_btn.click(
            fn=_on_install_whisperx,
            outputs=[diarize_model_dropdown, whisperx_status_text],
        )
        download_btn.click(
            fn=_on_download_model,
            inputs=[download_model_dropdown],
            outputs=[model_dropdown, download_model_dropdown],
        )
        delete_cache_btn.click(
            fn=_on_delete_cache,
            inputs=[download_model_dropdown],
            outputs=[model_dropdown, download_model_dropdown],
        )
        hf_token_input.change(
            fn=_on_save_token,
            inputs=[hf_token_input],
            outputs=[token_status],
        )
        output_dir_input.change(
            fn=_on_save_output_dir,
            inputs=[output_dir_input],
            outputs=[output_dir_status],
        )
        log_reload_btn.click(
            fn=_load_crash_log,
            outputs=[crash_log_text],
        )
        log_download_btn.click(
            fn=_on_download_crash_log,
            outputs=[log_download_output],
        )
        save_keywords_btn.click(
            fn=_on_save_keywords,
            inputs=[keywords_input],
        )
        keywords_input.select(
            fn=_on_select_keyword,
            inputs=[keywords_input],
            outputs=[keyword_edit_input, selected_keyword_idx],
        )
        keyword_add_btn.click(
            fn=_on_add_keyword,
            inputs=[keyword_edit_input, keywords_input, selected_keyword_idx],
            outputs=[keyword_edit_input, keywords_input, selected_keyword_idx],
        )
        keyword_edit_input.submit(
            fn=_on_add_keyword,
            inputs=[keyword_edit_input, keywords_input, selected_keyword_idx],
            outputs=[keyword_edit_input, keywords_input, selected_keyword_idx],
        )
        keyword_update_btn.click(
            fn=_on_update_keyword,
            inputs=[keyword_edit_input, keywords_input, selected_keyword_idx],
            outputs=[keyword_edit_input, keywords_input, selected_keyword_idx],
        )
        keyword_delete_btn.click(
            fn=_on_delete_keyword,
            inputs=[keywords_input, selected_keyword_idx],
            outputs=[keyword_edit_input, keywords_input, selected_keyword_idx],
        )
        export_keywords_btn.click(
            fn=_on_export_keywords,
            inputs=[keywords_input],
            outputs=[export_keywords_output],
        )
        import_keywords_file.change(
            fn=_on_import_keywords,
            inputs=[import_keywords_file],
            outputs=[keywords_input],
        )
        sort_kana_btn.click(
            fn=lambda df: _on_sort_keywords(df, "kana"),
            inputs=[keywords_input],
            outputs=[keywords_input],
        )
        sort_az_btn.click(
            fn=lambda df: _on_sort_keywords(df, "az"),
            inputs=[keywords_input],
            outputs=[keywords_input],
        )

        run_event = run_btn.click(
            fn=run,
            inputs=[
                file_input,
                mic_input,
                model_dropdown,
                num_speakers_input,
                hf_token_input,
                prompt_input,
                keywords_input,
                language_dropdown,
                gpu_dropdown,
                timestamp_toggle,
                diarization_toggle,
                diarize_model_dropdown,
                word_timestamps_toggle,
            ],
            outputs=[
                output_files, preview_text, model_info_text,
                segments_state, base_name_state, speaker_filter,
                history_dropdown,
                segments_table, seg_speaker_input,
            ],
        )
        history_load_btn.click(
            fn=_on_load_history,
            inputs=[history_dropdown, timestamp_toggle],
            outputs=[preview_text, segments_state, base_name_state, speaker_filter,
                     segments_table, seg_speaker_input],
        )
        history_delete_btn.click(
            fn=_on_delete_history,
            inputs=[history_dropdown],
            outputs=[history_dropdown],
        )
        replace_btn.click(
            fn=_on_replace,
            inputs=[
                segments_state, speaker_replace_input, text_replace_input,
                timestamp_toggle,
            ],
            outputs=[preview_text, segments_state, speaker_filter,
                     segments_table, seg_speaker_input],
        )
        speaker_filter.change(
            fn=_on_speaker_filter_change,
            inputs=[segments_state, speaker_filter, timestamp_toggle],
            outputs=[preview_text],
        )
        speaker_eq_btn.click(
            fn=_insert_equals,
            inputs=[speaker_replace_input],
            outputs=[speaker_replace_input],
        )
        text_eq_btn.click(
            fn=_insert_equals,
            inputs=[text_replace_input],
            outputs=[text_replace_input],
        )
        segments_table.select(
            fn=_on_select_segment,
            inputs=[segments_table],
            outputs=[segment_selected_idx, seg_speaker_input, seg_text_input,
                     seg_split_pos],
        )
        seg_update_btn.click(
            fn=_on_update_segment,
            inputs=[segments_state, segment_selected_idx,
                    seg_speaker_input, seg_text_input, timestamp_toggle],
            outputs=[segments_state, segments_table, preview_text,
                     speaker_filter],
        )
        seg_merge_btn.click(
            fn=_on_merge_segment,
            inputs=[segments_state, segment_selected_idx, timestamp_toggle],
            outputs=[segments_state, segments_table, preview_text,
                     segment_selected_idx, speaker_filter],
        )
        seg_split_btn.click(
            fn=_on_split_segment,
            inputs=[segments_state, segment_selected_idx,
                    seg_split_time, seg_split_pos, timestamp_toggle],
            outputs=[segments_state, segments_table, preview_text,
                     segment_selected_idx, speaker_filter],
        )
        seg_delete_btn.click(
            fn=_on_delete_segment,
            inputs=[segments_state, segment_selected_idx, timestamp_toggle],
            outputs=[segments_state, segments_table, preview_text,
                     segment_selected_idx, speaker_filter],
        )
        cancel_btn.click(
            fn=None,
            cancels=[run_event],
        )
        reexport_btn.click(
            fn=_on_reexport,
            inputs=[segments_state, preview_text, format_checkbox,
                    base_name_state, speaker_filter],
            outputs=[output_files],
        )
        timestamp_toggle.change(
            fn=_on_toggle_timestamp,
            inputs=[segments_state, timestamp_toggle],
            outputs=[preview_text],
        )
        diarization_toggle.change(
            fn=_on_toggle_diarization,
            inputs=[diarization_toggle],
            outputs=[diarize_model_dropdown, num_speakers_input],
        )
        # ブラウザを閉じたら自動終了（リロード猶予5秒）
        _shutdown_timer = {"timer": None}

        def _on_unload():
            if _shutdown_timer["timer"]:
                _shutdown_timer["timer"].cancel()
            _shutdown_timer["timer"] = threading.Timer(
                5.0, lambda: os.kill(os.getpid(), signal.SIGTERM)
            )
            _shutdown_timer["timer"].daemon = True
            _shutdown_timer["timer"].start()

        def _on_load():
            if _shutdown_timer["timer"]:
                _shutdown_timer["timer"].cancel()
                _shutdown_timer["timer"] = None

        app.unload(_on_unload)
        app.load(_on_load)

    return app


_GRADIO_THEME = gr.themes.Soft(
    font=["Noto Sans JP", "sans-serif"],
    font_mono=["Noto Sans Mono", "monospace"],
)
_GRADIO_CSS = (
    "@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap');"
    "footer {display: none !important;}"
    ".toast-wrap, .toast-body, .toast-text, .error, .error-text,"
    "[data-testid='error'] { user-select: text !important; -webkit-user-select: text !important; }"
)


def _show_error_dialog(message):
    """エラーダイアログを表示する（コンソールなし環境用）。"""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0, message, "Sasayaki - Error", 0x10,
        )
    except Exception:
        pass


if __name__ == "__main__":
    # pythonw.exe ではstderrが無いため crash.log にリダイレクト
    import sys
    _debug_mode = "--debug" in sys.argv
    if sys.stderr is None or not hasattr(sys.stderr, "write"):
        sys.stderr = open(CRASH_LOG_PATH, "w", encoding="utf-8")

    try:
        app = build_ui()

        try:
            import webview
        except ImportError:
            webview = None

        if webview:
            # pywebview: native window mode
            app.launch(
                inbrowser=False,
                prevent_thread_lock=True,
                show_error=True,
                theme=_GRADIO_THEME,
                css=_GRADIO_CSS,
            )
            _icon_path = os.path.join(APP_DIR, "sasayaki.ico")

            def _set_window_icon(icon_path):
                """Win32 API でウィンドウアイコンを設定する"""
                if sys.platform != "win32" or not os.path.exists(icon_path):
                    return
                try:
                    import ctypes
                    from ctypes import wintypes
                    user32 = ctypes.windll.user32
                    # アイコンをファイルから読み込み
                    IMAGE_ICON = 1
                    LR_LOADFROMFILE = 0x0010
                    LR_DEFAULTSIZE = 0x0040
                    icon_handle = user32.LoadImageW(
                        0, icon_path, IMAGE_ICON, 0, 0,
                        LR_LOADFROMFILE | LR_DEFAULTSIZE,
                    )
                    if not icon_handle:
                        return
                    # pywebview ウィンドウのハンドルを取得
                    hwnd = user32.FindWindowW(None, f"Sasayaki v{__version__}")
                    if not hwnd:
                        return
                    # WM_SETICON でアイコンを設定
                    WM_SETICON = 0x0080
                    ICON_SMALL = 0
                    ICON_BIG = 1
                    user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, icon_handle)
                    user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, icon_handle)
                except Exception:
                    pass  # アイコン設定失敗は無視

            webview.create_window(
                f"Sasayaki v{__version__}",
                "http://127.0.0.1:7860",
            )
            webview.start(func=_set_window_icon, args=[_icon_path],
                         debug=_debug_mode)
        else:
            # fallback: browser mode
            app.launch(
                inbrowser=True,
                show_error=True,
                theme=_GRADIO_THEME,
                css=_GRADIO_CSS,
            )
    except Exception:
        import traceback
        error_text = traceback.format_exc()
        with open(CRASH_LOG_PATH, "w", encoding="utf-8") as f:
            f.write(error_text)
        _show_error_dialog(
            f"Sasayaki failed to start.\n\n"
            f"Details saved to crash.log\n\n{error_text[:500]}"
        )
        sys.exit(1)
