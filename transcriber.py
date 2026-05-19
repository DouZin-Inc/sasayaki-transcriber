"""文字起こし＆話者分離のコアロジック"""

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time

import torch

from faster_whisper import WhisperModel


ALL_MODELS = ["tiny", "base", "small", "medium", "large-v3"]


def get_gpu_list():
    """利用可能なGPUの一覧を返す。[(表示名, デバイスID), ...]"""
    gpus = [("CPU", "cpu")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            gpus.insert(i, (f"GPU {i}: {name} ({vram:.0f}GB)", f"cuda:{i}"))
    return gpus

# モデルキャッシュ（同一セッション内でリロードを回避）
_whisper_cache = {"model_size": None, "device": None, "model": None}


def _validate_model_cache(model_size):
    """モデルキャッシュが有効か検証する。失敗時は自動削除して False を返す。"""
    # sasayaki 独自キャッシュを優先してパスを渡す
    local_dir = _get_local_model_dir(model_size)
    model_path = local_dir if os.path.isdir(local_dir) else model_size
    try:
        WhisperModel(model_path, device="cpu", compute_type="int8")
        return True
    except Exception:
        # キャッシュ破損 → 自動削除
        try:
            delete_model_cache(model_size)
        except Exception:
            pass
        return False


def _is_model_cached(model_size):
    """モデルがキャッシュに存在するか判定する。"""
    # 1. sasayaki 独自キャッシュ
    local_dir = _get_local_model_dir(model_size)
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        return True
    # 2. HF キャッシュ（従来の場所）
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        snapshots = os.path.join(
            HF_HUB_CACHE,
            f"models--Systran--faster-whisper-{model_size}",
            "snapshots",
        )
        if os.path.isdir(snapshots) and os.listdir(snapshots):
            return True
    except ImportError:
        pass
    return False


def get_downloaded_models(verify=False):
    """ダウンロード済みのWhisperモデル一覧を返す。

    Args:
        verify: True の場合、モデルの整合性チェックを行い、
                破損したキャッシュは自動削除する。
    """
    downloaded = []
    for model_size in ALL_MODELS:
        if _is_model_cached(model_size):
            if verify:
                if _validate_model_cache(model_size):
                    downloaded.append(model_size)
                else:
                    print(
                        f"[WARN] Model {model_size} cache is corrupted, "
                        f"removed automatically.",
                        flush=True,
                    )
            else:
                downloaded.append(model_size)
    return downloaded


def _rm_readonly(func, path, _exc_info):
    """shutil.rmtree 用: 読み取り専用ファイルを削除可能にする。"""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _get_local_model_dir(model_size):
    """symlink を使わないローカルモデルディレクトリのパスを返す。"""
    return os.path.join(
        os.path.expanduser("~"), ".cache", "sasayaki", "models",
        f"Systran--faster-whisper-{model_size}",
    )


def _get_model_cache_dir(model_size):
    """モデルキャッシュのディレクトリパスを返す。"""
    # 1. sasayaki 独自キャッシュ（symlink 回避）
    local_dir = _get_local_model_dir(model_size)
    if os.path.isdir(local_dir):
        return local_dir
    # 2. huggingface_hub の定数から取得
    dir_name = f"models--Systran--faster-whisper-{model_size}"
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        path = os.path.join(HF_HUB_CACHE, dir_name)
        if os.path.isdir(path):
            return path
    except ImportError:
        pass
    # 3. 環境変数からフォールバック
    for env in ("HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        env_val = os.environ.get(env)
        if env_val:
            path = os.path.join(env_val, "hub", dir_name) if env == "HF_HOME" else os.path.join(env_val, dir_name)
            if os.path.isdir(path):
                return path
    # 4. デフォルトパス (~/.cache/huggingface/hub)
    path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", dir_name)
    if os.path.isdir(path):
        return path
    return None


def delete_model_cache(model_size):
    """指定モデルのキャッシュを削除する（sasayaki 独自 + HF キャッシュ両方）。"""
    deleted_any = False
    # 1. sasayaki 独自キャッシュ
    local_dir = _get_local_model_dir(model_size)
    if os.path.isdir(local_dir):
        shutil.rmtree(local_dir, onerror=_rm_readonly)
        deleted_any = True
    # 2. HF キャッシュ（従来の場所）
    dir_name = f"models--Systran--faster-whisper-{model_size}"
    hf_cache_dir = None
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        path = os.path.join(HF_HUB_CACHE, dir_name)
        if os.path.isdir(path):
            hf_cache_dir = path
    except ImportError:
        pass
    if not hf_cache_dir:
        path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", dir_name)
        if os.path.isdir(path):
            hf_cache_dir = path
    if hf_cache_dir:
        shutil.rmtree(hf_cache_dir, onerror=_rm_readonly)
        deleted_any = True
    # メモリ上のキャッシュを無条件クリア＆GPUメモリ解放
    _whisper_cache["model"] = None
    _whisper_cache["model_size"] = None
    _whisper_cache["device"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_progress_tqdm(progress_callback, model_size):
    """ダウンロード進捗をコールバックに転送する tqdm クラスを生成する。"""
    if not progress_callback:
        return None
    try:
        from tqdm.auto import tqdm as _base_tqdm
    except ImportError:
        return None

    _cb = progress_callback
    _ms = model_size
    _last_report = [0.0]

    class _ProgressTqdm(_base_tqdm):
        def update(self, n=1):
            super().update(n)
            if self.total and self.total > 1_000_000:
                now = time.time()
                if now - _last_report[0] < 0.5:
                    return
                _last_report[0] = now
                ratio = min(self.n / self.total, 1.0)
                mb_done = self.n / (1024 ** 2)
                mb_total = self.total / (1024 ** 2)
                _cb(ratio * 0.9, f"ダウンロード中 ({_ms})... {mb_done:.0f}/{mb_total:.0f} MB")

    return _ProgressTqdm


def download_model(model_size, progress_callback=None):
    """Whisperモデルを事前にダウンロード（キャッシュ）する。

    local_dir を指定して実ファイルとしてダウンロードする（symlink 回避）。
    2回目以降はキャッシュから読み込むため即座に完了する。
    """
    repo_id = f"Systran/faster-whisper-{model_size}"

    if progress_callback:
        progress_callback(0.0, f"モデル {model_size} をダウンロード中...")

    tqdm_class = _build_progress_tqdm(progress_callback, model_size)

    try:
        model_path = _download_hf_model(repo_id, model_size, tqdm_class)
    except ImportError:
        model_path = model_size

    if progress_callback:
        progress_callback(0.95, f"モデルを検証中...")

    try:
        WhisperModel(model_path, device="cpu", compute_type="int8")
    except RuntimeError as e:
        if "Unable to open file" in str(e):
            print(f"[WARN] モデル読み込み失敗、キャッシュを削除して再ダウンロードします: {e}", flush=True)
            delete_model_cache(model_size)
            try:
                model_path = _download_hf_model(repo_id, model_size, tqdm_class, force=True)
            except ImportError:
                model_path = model_size
            WhisperModel(model_path, device="cpu", compute_type="int8")
        else:
            raise

    if progress_callback:
        progress_callback(1.0, f"モデル {model_size} のダウンロードが完了しました")


def _download_hf_model(repo_id, model_size, tqdm_class=None, force=False):
    """huggingface_hub API でモデルを local_dir にダウンロードし、パスを返す。

    Windows の symlink 問題を回避するため、HF キャッシュの symlink 構造を使わず
    local_dir に実ファイルとして直接ダウンロードする。
    """
    from huggingface_hub import snapshot_download

    local_dir = _get_local_model_dir(model_size)

    dl_kwargs = {"local_dir": local_dir}
    if force:
        dl_kwargs["force_download"] = True

    # tqdm をモンキーパッチして進捗コールバックを差し込む
    _patches = []  # [(module, attr, original_value), ...]
    if tqdm_class:
        try:
            import tqdm.auto as _tqdm_mod
            _patches.append((_tqdm_mod, "tqdm", _tqdm_mod.tqdm))
            _tqdm_mod.tqdm = tqdm_class
        except Exception:
            pass
        try:
            from huggingface_hub import file_download as _hf_fd
            if hasattr(_hf_fd, "tqdm"):
                _patches.append((_hf_fd, "tqdm", _hf_fd.tqdm))
                _hf_fd.tqdm = tqdm_class
        except Exception:
            pass

    try:
        snapshot_download(repo_id, **dl_kwargs)
    finally:
        for mod, attr, orig in _patches:
            setattr(mod, attr, orig)

    return local_dir


DIARIZE_MODELS = [
    ("3.1", "pyannote/speaker-diarization-3.1"),
    ("community-1（対応予定）", "pyannote/speaker-diarization-community-1"),
    ("WhisperX", "whisperx"),
    ("NeMo（対応予定）", "nemo"),
]

# 現在利用可能なモデルの値セット
_SUPPORTED_DIARIZE_MODELS = {
    "pyannote/speaker-diarization-3.1",
    "whisperx",
}


def is_whisperx_available():
    """WhisperX がインストールされているか確認する。"""
    try:
        import whisperx  # noqa: F401
        return True
    except ImportError:
        return False


def _strip_prompt_leak(segments, initial_prompt):
    """Whisper が initial_prompt をハルシネーション出力した場合に除去する。

    Whisper はセグメント境界や無音区間で initial_prompt のテキストを
    そのまま（または微妙に変えて）出力することがある。
    1. 先頭一致でループ除去（繰り返しハルシネーション対応）
    2. 残テキストがプロンプトと高類似なら丸ごと除去（微妙な変形対応）
    """
    if not initial_prompt or not segments:
        return segments

    from difflib import SequenceMatcher

    # 除去対象: フルプロンプトと、その構成パーツ（空白で分割）
    targets = [initial_prompt]
    for part in initial_prompt.split():
        if part and part != initial_prompt:
            targets.append(part)
    # 長い順にマッチさせる（フルプロンプトを先に試す）
    targets.sort(key=len, reverse=True)

    cleaned = []
    for seg in segments:
        text = seg["text"].strip()
        # 先頭からプロンプト文字列をループ除去（繰り返しハルシネーション対応）
        changed = True
        while changed:
            changed = False
            for t in targets:
                if text.startswith(t):
                    text = text[len(t):].strip()
                    changed = True
        # 残テキストがプロンプトと高類似（Whisper が微妙に変えて出力した場合）→ 除去
        if text:
            best_ratio = max(
                SequenceMatcher(None, text, t).ratio()
                for t in targets
            )
            if best_ratio > 0.5:
                text = ""
        if text:
            cleaned.append({**seg, "text": text})
    return cleaned


def transcribe(audio_path, model_size="large-v3", num_speakers=2,
               hf_token=None, initial_prompt=None, language="ja",
               gpu_device="auto", progress_callback=None,
               enable_diarization=True,
               diarize_model="pyannote/speaker-diarization-3.1",
               word_timestamps=False):
    """動画/音声ファイルを文字起こし（＋オプションで話者分離）する。

    Args:
        audio_path: 入力ファイルのパス
        model_size: Whisperモデルサイズ (tiny/base/small/medium/large-v3)
        num_speakers: 話者数
        hf_token: HuggingFace APIトークン（pyannote用）
        progress_callback: 進捗通知用コールバック fn(progress: float, message: str)
        enable_diarization: 話者分離を有効にするか
        word_timestamps: 単語単位のタイムスタンプでセグメント境界を補正する

    Returns:
        list[dict]: セグメントのリスト
            各セグメント: {"start": float, "end": float, "text": str, "speaker": str}
    """

    def _progress(value, msg):
        print(f"[{value:.0%}] {msg}", flush=True)
        if progress_callback:
            progress_callback(value, msg)

    # --- 1. 音声抽出（WAV変換） ---
    _progress(0.0, "音声を抽出しています...")
    wav_path = _extract_audio(audio_path)

    try:
        # WhisperX パイプライン: 文字起こし・アライメント・話者分離を一括実行
        if enable_diarization and diarize_model == "whisperx":
            _progress(0.1, "WhisperX パイプラインを起動中...")
            segments = _run_whisperx(
                wav_path, model_size, language, num_speakers,
                hf_token, gpu_device, initial_prompt, _progress,
            )
            _progress(1.0, "完了しました")
            return _strip_prompt_leak(segments, initial_prompt)

        # --- 2. 文字起こし（faster-whisper） ---
        _progress(0.1, f"モデルを読み込み中（{model_size}）...")
        segments_raw = _run_whisper(wav_path, model_size, initial_prompt, language, _progress, gpu_device, word_timestamps)
        print(f"[STEP] Whisper完了（{len(segments_raw)}セグメント）", flush=True)

        if enable_diarization:
            # --- 3. 話者分離（pyannote） ---
            _progress(0.6, "話者分離パイプラインを準備中...")
            print("[STEP] 話者分離パイプライン読み込み開始", flush=True)
            diarization = _run_diarization(wav_path, num_speakers, hf_token, gpu_device, _progress, diarize_model)
            print("[STEP] 話者分離完了", flush=True)

            # --- 4. マージ ---
            _progress(0.9, "結果を統合しています...")
            segments = _merge(segments_raw, diarization)
        else:
            # 話者分離スキップ — 全セグメントに「話者1」を設定
            segments = []
            for seg in segments_raw:
                segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "speaker": "話者1",
                })

        _progress(1.0, "完了しました")
        return _strip_prompt_leak(segments, initial_prompt)

    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            "GPU メモリ（VRAM）が不足しています。\n"
            "より小さいモデル（small / base / tiny）を選択してください。\n"
            "または、他のアプリケーションを閉じて VRAM を解放してください。"
        )
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def _extract_audio(input_path):
    """ffmpegで入力ファイルをモノラル16kHz WAVに変換する。"""
    wav_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000", "-vn",
        wav_path,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg でエラーが発生しました:\n{result.stderr}"
        )
    return wav_path


def _parse_device(gpu_device):
    """gpu_device文字列をdevice名とインデックスに分解する。"""
    if gpu_device == "auto":
        return ("cuda" if torch.cuda.is_available() else "cpu"), 0
    if gpu_device == "cpu":
        return "cpu", 0
    if ":" in gpu_device:
        dev, idx = gpu_device.split(":", 1)
        return dev, int(idx)
    return gpu_device, 0


def _get_whisper_model(model_size, gpu_device="auto"):
    """Whisperモデルをキャッシュから取得、または新規ロードする。"""
    device, device_index = _parse_device(gpu_device)
    compute_type = "float16" if device != "cpu" else "int8"

    if (_whisper_cache["model"] is not None
            and _whisper_cache["model_size"] == model_size
            and _whisper_cache["device"] == gpu_device):
        return _whisper_cache["model"]

    # sasayaki 独自キャッシュがあればそちらを優先（symlink 回避）
    local_dir = _get_local_model_dir(model_size)
    model_id = local_dir if os.path.isdir(local_dir) else model_size
    try:
        model = WhisperModel(model_id, device=device, device_index=device_index, compute_type=compute_type)
    except RuntimeError as e:
        if "Unable to open file" in str(e):
            raise RuntimeError(
                f"モデル {model_size} のキャッシュが破損しています。\n"
                "設定タブから再ダウンロードしてください。"
            ) from None
        raise
    _whisper_cache["model_size"] = model_size
    _whisper_cache["device"] = gpu_device
    _whisper_cache["model"] = model
    return model


def _release_whisper_model():
    """Whisperモデルを解放してGPUメモリを空ける。"""
    try:
        _whisper_cache["model"] = None
        _whisper_cache["model_size"] = None
        _whisper_cache["device"] = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                vram_free = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                print(f"[GPU] Whisperモデル解放後 空きVRAM: {vram_free:.1f}GB", flush=True)
            except Exception:
                print("[GPU] Whisperモデル解放完了（VRAM情報取得不可）", flush=True)
        else:
            print("[CPU] Whisperモデル解放完了", flush=True)
    except Exception as e:
        print(f"[警告] Whisperモデル解放中にエラー: {e}", flush=True)


def _run_whisper(wav_path, model_size, initial_prompt=None, language="ja", progress_fn=None, gpu_device="auto", word_timestamps=False):
    """faster-whisperで文字起こしを実行し、セグメントリストを返す。"""
    model = _get_whisper_model(model_size, gpu_device)
    if progress_fn:
        progress_fn(0.2, "文字起こし中... 動画の長さに応じて時間がかかります")
    segments_iter, info = model.transcribe(
        wav_path,
        beam_size=5,
        language=language or None,
        vad_filter=True,
        initial_prompt=initial_prompt,
        word_timestamps=word_timestamps,
    )

    duration = info.duration or 1.0
    segments = []
    import gc
    gc.disable()  # Whisper反復中のGCを抑止（CTranslate2 segfault回避）
    try:
        for seg in segments_iter:
            start = seg.start
            end = seg.end
            # word_timestamps有効時: 単語レベルのタイムスタンプでセグメント境界を補正
            if word_timestamps and seg.words:
                start = seg.words[0].start
                end = seg.words[-1].end
            segments.append({
                "start": start,
                "end": end,
                "text": seg.text.strip(),
            })
            if progress_fn:
                # 文字起こしフェーズは 0.2〜0.5 の範囲で進捗表示
                ratio = min(seg.end / duration, 1.0)
                progress_fn(0.2 + ratio * 0.3, f"文字起こし中... {seg.end:.0f}/{duration:.0f}秒")
    finally:
        gc.enable()
    print(f"[INFO] 文字起こし完了: {len(segments)}セグメント", flush=True)
    return segments


def _kill_process_tree(proc):
    """サブプロセスとその子プロセスを全て終了させる。"""
    if proc.poll() is not None:
        return  # 既に終了済み
    try:
        if sys.platform == "win32":
            # Windows: taskkill /T でプロセスツリーごと強制終了
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            proc.kill()
        proc.wait(timeout=10)
    except Exception as e:
        print(f"[Diarization] プロセス終了エラー: {e}", flush=True)


def _get_python_exe():
    """サブプロセス用のPython実行ファイルパスを返す。

    embed版Pythonの場合 sys.executable がスクリプトディレクトリの python/python.exe を
    返さないケースがあるため、フォールバックを用意する。
    """
    if sys.executable and os.path.isfile(sys.executable):
        return sys.executable
    # embed版フォールバック: スクリプトと同階層の python/python.exe
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fallback = os.path.join(script_dir, "python", "python.exe")
    if os.path.isfile(fallback):
        return fallback
    raise RuntimeError(
        "Python 実行ファイルが見つかりません。\n"
        f"sys.executable={sys.executable!r}\n"
        f"フォールバック={fallback!r}"
    )


def _run_diarization(wav_path, num_speakers, hf_token, gpu_device="auto", progress_fn=None,
                     diarize_model="pyannote/speaker-diarization-3.1"):
    """話者分離をサブプロセスで実行する。

    diarize_worker.py を別プロセスとして起動し、CUDA コンテキストの競合を回避する。
    stderr から進捗を読み取り、stdout から JSON 結果を取得する。
    """
    if not hf_token:
        raise ValueError(
            "話者分離には HuggingFace トークンが必要です。\n"
            "設定画面でトークンを入力してください。"
        )

    python_exe = _get_python_exe()
    worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diarize_worker.py")

    cmd = [
        python_exe, worker_script,
        "--wav-path", wav_path,
        "--hf-token", hf_token,
        "--gpu-device", gpu_device,
        "--model", diarize_model,
    ]
    if num_speakers > 0:
        cmd.extend(["--num-speakers", str(num_speakers)])

    print(f"[Diarization] サブプロセス起動: {python_exe}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    # stderr を別スレッドで読み取る（stdout/stderr 同時読み取りでパイプデッドロック回避）
    error_messages = []

    def _read_stderr():
        for line in proc.stderr:
            line = line.rstrip("\n")
            if line.startswith("PROGRESS:"):
                parts = line.split(":", 2)
                if len(parts) == 3:
                    try:
                        value = float(parts[1])
                        msg = parts[2]
                        print(f"[Diarization] {value:.0%} {msg}", flush=True)
                        if progress_fn:
                            progress_fn(value, f"話者分離中... {msg}")
                    except ValueError:
                        pass
            elif line.startswith("ERROR:"):
                error_messages.append(line[6:])
            elif line.strip():
                print(f"[Diarization:stderr] {line}", flush=True)

    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

    try:
        stdout_data = proc.stdout.read()
        proc.wait()
        stderr_thread.join(timeout=10)
    except BaseException:
        # キャンセル等で中断された場合、サブプロセスを確実に終了させる
        print("[Diarization] 中断検出 — サブプロセスを終了します", flush=True)
        _kill_process_tree(proc)
        raise

    if proc.returncode != 0:
        error_detail = "\n".join(error_messages) if error_messages else "不明なエラー"
        raise RuntimeError(f"話者分離でエラーが発生しました:\n{error_detail}")

    # stdout から JSON 結果をパース
    try:
        diar_segments = json.loads(stdout_data)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"話者分離の結果解析に失敗しました:\n{e}\n"
            f"stdout: {stdout_data[:500]}"
        ) from e

    print(f"[Diarization] 完了: {len(diar_segments)}セグメント", flush=True)
    return diar_segments


def _merge(whisper_segments, diar_segments):
    """文字起こしセグメントに話者ラベルを付与する。

    各Whisperセグメントの時間範囲と最も重なるdiarizationの話者を割り当てる。

    Args:
        whisper_segments: Whisperの文字起こしセグメント
        diar_segments: 話者分離結果 [{"start", "end", "speaker"}, ...]
    """
    result = []
    for seg in whisper_segments:
        speaker = _find_speaker(seg["start"], seg["end"], diar_segments)
        result.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "speaker": speaker,
        })
    return result


def _find_speaker(start, end, diar_segments):
    """指定した時間範囲で最も重複が大きい話者を返す。"""
    best_speaker = "不明"
    best_overlap = 0.0

    for dseg in diar_segments:
        overlap_start = max(start, dseg["start"])
        overlap_end = min(end, dseg["end"])
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = dseg["speaker"]

    return best_speaker


def _run_whisperx(wav_path, model_size, language, num_speakers,
                  hf_token, gpu_device, initial_prompt, progress_fn=None):
    """WhisperX パイプラインをサブプロセスで実行する。

    whisperx_worker.py を別プロセスとして起動し、
    文字起こし・アライメント・話者分離を一括で実行する。
    """
    if not hf_token:
        raise ValueError(
            "WhisperX の話者分離には HuggingFace トークンが必要です。\n"
            "設定画面でトークンを入力してください。"
        )

    python_exe = _get_python_exe()
    worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisperx_worker.py")

    # sasayaki 独自キャッシュがあればパスを渡す（HF symlink 回避）
    local_dir = _get_local_model_dir(model_size)
    model_id = local_dir if os.path.isdir(local_dir) else model_size

    cmd = [
        python_exe, worker_script,
        "--wav-path", wav_path,
        "--model-size", model_id,
        "--language", language or "",
        "--hf-token", hf_token,
        "--gpu-device", gpu_device,
    ]
    if num_speakers > 0:
        cmd.extend(["--num-speakers", str(num_speakers)])
    if initial_prompt:
        cmd.extend(["--initial-prompt", initial_prompt])

    print(f"[WhisperX] サブプロセス起動: {python_exe}", flush=True)

    # Whisperモデルを解放してGPUメモリを空ける
    _release_whisper_model()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    error_messages = []
    stderr_lines = []

    def _read_stderr():
        for line in proc.stderr:
            line = line.rstrip("\n")
            if line.startswith("PROGRESS:"):
                parts = line.split(":", 2)
                if len(parts) == 3:
                    try:
                        value = float(parts[1])
                        msg = parts[2]
                        print(f"[WhisperX] {value:.0%} {msg}", flush=True)
                        if progress_fn:
                            progress_fn(value, f"WhisperX: {msg}")
                    except ValueError:
                        pass
            elif line.startswith("ERROR:"):
                error_messages.append(line[6:])
            elif line.strip():
                stderr_lines.append(line)
                print(f"[WhisperX:stderr] {line}", flush=True)

    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

    try:
        stdout_data = proc.stdout.read()
        proc.wait()
        stderr_thread.join(timeout=10)
    except BaseException:
        print("[WhisperX] 中断検出 — サブプロセスを終了します", flush=True)
        _kill_process_tree(proc)
        raise

    if proc.returncode != 0:
        if error_messages:
            error_detail = "\n".join(error_messages)
        else:
            # 警告・プログレスバーを除外して実際のエラーだけ抽出
            _noise_patterns = (
                "Lightning automatically upgraded",
                "ReproducibilityWarning",
                "TensorFloat-32",
                "torch.backends.cuda",
                "torch.backends.cudnn",
                "pyannote/pyannote-audio",
                "warnings.warn",
                "Fetching",
                "it/s]",
                "it/s)",
            )
            real_errors = [
                l for l in stderr_lines
                if not any(p in l for p in _noise_patterns)
            ]
            if real_errors:
                error_detail = "\n".join(real_errors[-20:])
            elif stderr_lines:
                error_detail = (
                    f"サブプロセスがクラッシュしました (exit code {proc.returncode})。\n"
                    "GPU メモリ不足またはモデルダウンロードの失敗が考えられます。\n"
                    "「設定」タブで CPU に切り替えるか、より小さいモデルをお試しください。"
                )
            else:
                error_detail = f"不明なエラー (exit code {proc.returncode})"
        raise RuntimeError(f"WhisperX でエラーが発生しました:\n{error_detail}")

    try:
        segments = json.loads(stdout_data)
    except json.JSONDecodeError:
        # whisperx/pyannote のログが stdout に混入した場合、最終行の JSON を抽出
        last_line = stdout_data.strip().rsplit("\n", 1)[-1]
        try:
            segments = json.loads(last_line)
        except json.JSONDecodeError as e2:
            raise RuntimeError(
                f"WhisperX の結果解析に失敗しました:\n{e2}\n"
                f"stdout: {stdout_data[:500]}"
            ) from e2

    print(f"[WhisperX] 完了: {len(segments)}セグメント", flush=True)
    return segments
