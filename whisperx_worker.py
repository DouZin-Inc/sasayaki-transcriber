"""WhisperX パイプライン サブプロセスワーカー

transcriber.py から subprocess で起動される。
WhisperX は文字起こし・アライメント・話者分離を一括で実行するため、
既存の faster-whisper + pyannote のパイプラインとは独立して動作する。

引数:
    --wav-path       入力WAVファイルパス
    --model-size     Whisperモデルサイズ (tiny/base/small/medium/large-v3)
    --language       言語コード
    --num-speakers   話者数（0で自動推定）
    --hf-token       HuggingFace APIトークン
    --gpu-device     GPUデバイス指定 (auto/cpu/cuda:0 など)
    --initial-prompt 初期プロンプト（任意）

出力:
    stdout: JSON配列 [{"start": float, "end": float, "text": str, "speaker": str}, ...]
    stderr: PROGRESS:<value>:<message> 形式の進捗通知 / ERROR:<message> 形式のエラー
"""

import argparse
import inspect
import io
import json
import os
import sys

# HuggingFace / tqdm のプログレスバーを抑制（stderr パース干渉防止）
# 他モジュールの import 前に設定する必要がある
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

# pyannote / HuggingFace Hub のキャッシュをアプリローカルにリダイレクト
# pyannote は hf_hub_download(cache_dir=...) でサブモデルをダウンロードするため、
# TORCH_HOME だけでなく HF_HUB_CACHE も設定する必要がある。
# ※ import torch / huggingface_hub より前に設定すること
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_HF_CACHE = os.path.join(_APP_DIR, "models", "hf_cache")
os.environ["TORCH_HOME"] = os.path.join(_APP_DIR, "models", "torch_home")
os.environ["HF_HUB_CACHE"] = _APP_HF_CACHE

# Windows環境でUTF-8出力を保証（cp932文字化け回避）
# ※ 他の stderr 出力より先に設定する必要がある
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", newline="\n")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", newline="\n")

# stdout は JSON 出力専用。ライブラリ (whisperx/pyannote) が logging や
# warnings.warn() で stdout にログを書き込むと JSON パースが壊れるため、
# すべてのログ出力を stderr に固定する。ライブラリの import 前に設定すること。
import logging as _logging
_logging.basicConfig(stream=sys.stderr, force=True)
_logging.captureWarnings(True)

# Windows: シンボリックリンク作成失敗時にファイルコピーにフォールバック
# Developer Mode が無効な Windows では os.symlink が失敗し、
# HuggingFace / PyTorch のキャッシュが壊れる問題を根本回避する。
if sys.platform == "win32":
    import shutil as _shutil
    _original_symlink = os.symlink
    def _copy_instead_of_symlink(src, dst, target_is_directory=False):
        try:
            _original_symlink(src, dst, target_is_directory)
        except OSError:
            if os.path.isdir(src):
                _shutil.copytree(src, dst)
            else:
                _shutil.copy2(src, dst)
    os.symlink = _copy_instead_of_symlink

    # 壊れたキャッシュを自動検出・削除
    # ※ os.walk() は Windows の壊れたシンボリックリンクをスキップするため
    #    os.listdir() ベースの再帰関数を使用する

    def _has_broken_files(directory):
        """ディレクトリ内に壊れたシンボリックリンクや LFS ポインターファイルがないか再帰チェック。"""
        try:
            entries = os.listdir(directory)
        except OSError:
            return True
        for name in entries:
            fpath = os.path.join(directory, name)
            if os.path.isdir(fpath) and not os.path.islink(fpath):
                if _has_broken_files(fpath):
                    return True
                continue
            try:
                with open(fpath, "rb") as _fh:
                    head = _fh.read(64)
                if head.startswith(b"version https://git-lfs"):
                    return True
            except OSError:
                return True
        return False

    def _rmtree_force(path):
        """shutil.rmtree が扱えない壊れたシンボリックリンクも含めて削除する。"""
        _remove_all_entries(path)
        if os.path.isdir(path):
            _shutil.rmtree(path, ignore_errors=True)

    def _force_delete_entry(path):
        """壊れたシンボリックリンク/ジャンクションも含めて1エントリを削除する。"""
        for fn in (os.unlink, os.rmdir):
            try:
                fn(path)
                return
            except OSError:
                pass
        # Python API で削除できない場合、Windows API を直接呼び出す
        try:
            import ctypes as _ct
            k32 = _ct.windll.kernel32
            if not k32.DeleteFileW(path):
                k32.RemoveDirectoryW(path)
        except Exception:
            pass

    def _is_reparse_point(path):
        """NTFS リパースポイント（シンボリックリンク/ジャンクション）を検出する。"""
        try:
            import ctypes as _ct
            attrs = _ct.windll.kernel32.GetFileAttributesW(path)
            return attrs != -1 and bool(attrs & 0x0400)
        except Exception:
            return False

    def _remove_all_entries(directory):
        """os.listdir ベースの再帰削除（壊れたリンク/ジャンクションも対応）。"""
        try:
            entries = os.listdir(directory)
        except OSError:
            return
        for name in entries:
            fpath = os.path.join(directory, name)
            # リパースポイント（シンボリックリンク/ジャンクション）は中に入らず直接削除
            if _is_reparse_point(fpath):
                _force_delete_entry(fpath)
            elif os.path.isdir(fpath):
                _remove_all_entries(fpath)
                try:
                    os.rmdir(fpath)
                except OSError:
                    pass
            else:
                _force_delete_entry(fpath)

    def _cleanup_broken_cache():
        _cache_roots = [
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
            os.path.join(os.path.expanduser("~"), ".cache", "torch", "pyannote"),
            _APP_HF_CACHE,
            os.path.join(_APP_DIR, "models", "torch_home", "pyannote"),
        ]
        for cache_root in _cache_roots:
            if not os.path.isdir(cache_root):
                continue
            for entry in os.listdir(cache_root):
                if not entry.startswith("models--"):
                    continue
                model_dir = os.path.join(cache_root, entry)
                snapshots_dir = os.path.join(model_dir, "snapshots")
                if not os.path.isdir(snapshots_dir):
                    continue
                if _has_broken_files(snapshots_dir):
                    _rmtree_force(model_dir)
                    print(f"INFO: broken cache deleted: {model_dir}",
                          file=sys.stderr, flush=True)
    try:
        _cleanup_broken_cache()
    except Exception as e:
        print(f"WARNING: cache cleanup failed: {e}", file=sys.stderr, flush=True)

import torch

# --- monkey patches (サブプロセス側のみで必要) ---

# pyannote v3.x は import 時に torch.hub._get_torch_home() でキャッシュパスを
# 固定するが、TORCH_HOME 環境変数を無視する場合がある。関数を直接差し替える。
torch.hub._get_torch_home = lambda: os.environ["TORCH_HOME"]

# pyannote v3.x が内部で hf_hub_download(cache_dir=...) でサブモデルをダウンロードする。
# cache_dir モードは blobs/snapshots 間のシンボリックリンクを生成し、
# Developer Mode が無効な Windows では壊れたリンクになる。
# → cache_dir を local_dir に差し替え、シンボリックリンクなしで直接ダウンロードさせる。
import huggingface_hub as _hf_hub
_original_hf_hub_download = _hf_hub.hf_hub_download
_default_torch_cache_norm = os.path.normpath(
    os.path.join(os.path.expanduser("~"), ".cache", "torch")
)
_redirect_cache_prefixes = [
    _default_torch_cache_norm,
    os.path.normpath(_APP_HF_CACHE),
    os.path.normpath(os.path.join(os.environ["TORCH_HOME"], "pyannote")),
]
def _patched_hf_hub_download(*args, **kwargs):
    cache_dir = kwargs.get("cache_dir")
    if cache_dir:
        norm = os.path.normpath(str(cache_dir))
        if any(norm.startswith(p) for p in _redirect_cache_prefixes):
            repo_id = args[0] if args else kwargs.get("repo_id", "unknown")
            safe_name = repo_id.replace("/", "--")
            kwargs.pop("cache_dir", None)
            kwargs["local_dir"] = os.path.join(_APP_HF_CACHE, safe_name)
    return _original_hf_hub_download(*args, **kwargs)
_hf_hub.hf_hub_download = _patched_hf_hub_download

# PyTorch 2.6: weights_only=True がデフォルトだが pyannote 3.x 非対応
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# speechbrain の inspect.getframeinfo が embed版Python で無限再帰するため回避
_original_getframeinfo = inspect.getframeinfo
def _safe_getframeinfo(*args, **kwargs):
    try:
        return _original_getframeinfo(*args, **kwargs)
    except RecursionError:
        return inspect.Traceback("<unknown>", 0, "<unknown>", None, None)
inspect.getframeinfo = _safe_getframeinfo

# pyannote v3/v4 互換パッチ: token vs use_auth_token
try:
    from pyannote.audio import Pipeline as _OrigPipeline
    _orig_from_pretrained = _OrigPipeline.from_pretrained
    _fp_params = inspect.signature(_orig_from_pretrained).parameters
    if "token" not in _fp_params and "use_auth_token" in _fp_params:
        # pyannote v3: token → use_auth_token に変換
        @staticmethod
        def _patched_from_pretrained(*args, **kwargs):
            token = kwargs.pop("token", None)
            if token is not None:
                kwargs["use_auth_token"] = token
            return _orig_from_pretrained(*args, **kwargs)
        _OrigPipeline.from_pretrained = _patched_from_pretrained
except Exception:
    pass

# pyannote の Inference が token パラメータを受け付けないバージョンへの対応
try:
    from pyannote.audio.core.inference import Inference as _OrigInference
    _orig_inf_init = _OrigInference.__init__
    _inf_params = inspect.signature(_orig_inf_init).parameters
    if "token" not in _inf_params:
        def _patched_inf_init(self, *args, **kwargs):
            kwargs.pop("token", None)
            return _orig_inf_init(self, *args, **kwargs)
        _OrigInference.__init__ = _patched_inf_init
except Exception:
    pass


def _progress(value, message):
    """進捗を stderr に出力する。"""
    print(f"PROGRESS:{value:.2f}:{message}", file=sys.stderr, flush=True)


def _error(message):
    """エラーを stderr に出力する。改行は ' | ' に置換して1行に収める。"""
    safe = str(message).replace("\r", "").replace("\n", " | ")
    print(f"ERROR:{safe}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description="WhisperX パイプラインワーカー")
    parser.add_argument("--wav-path", required=True, help="入力WAVファイルパス")
    parser.add_argument("--model-size", default="large-v3", help="Whisperモデルサイズ")
    parser.add_argument("--language", default="ja", help="言語コード")
    parser.add_argument("--num-speakers", type=int, default=0, help="話者数（0で自動推定）")
    parser.add_argument("--hf-token", required=True, help="HuggingFace APIトークン")
    parser.add_argument("--gpu-device", default="auto", help="GPUデバイス (auto/cpu/cuda:0)")
    parser.add_argument("--initial-prompt", default=None, help="初期プロンプト")
    args = parser.parse_args()

    try:
        # HF トークンを環境変数に設定（alignment モデルの from_pretrained 等が自動参照）
        if args.hf_token:
            os.environ["HF_TOKEN"] = args.hf_token

        import whisperx

        # デバイス決定
        if args.gpu_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            device_index = 0
        elif args.gpu_device == "cpu":
            device = "cpu"
            device_index = 0
        elif args.gpu_device.startswith("cuda:"):
            device = "cuda"
            device_index = int(args.gpu_device.split(":")[1])
        else:
            device = args.gpu_device
            device_index = 0
        compute_type = "float16" if device != "cpu" else "int8"

        # --- 1. 文字起こし ---
        _progress(0.10, "WhisperX モデルを読み込み中...")
        load_opts = {}
        if args.initial_prompt:
            load_opts["asr_options"] = {"initial_prompt": args.initial_prompt}
        model = whisperx.load_model(
            args.model_size,
            device,
            device_index=device_index,
            compute_type=compute_type,
            **load_opts,
        )

        _progress(0.20, "文字起こし中...")
        audio = whisperx.load_audio(args.wav_path)
        transcribe_opts = {}
        if args.language:
            transcribe_opts["language"] = args.language
        result = model.transcribe(audio, **transcribe_opts)
        detected_language = result.get("language", args.language or "ja")

        # モデルを解放してメモリを空ける
        del model
        import gc
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

        _progress(0.40, "アライメント中...")

        # --- 2. アライメント ---
        # アライメントモデルのロードに失敗する環境があるため、
        # 失敗時はアライメントをスキップして文字起こし結果をそのまま使う。
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=device,
            )
            result = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            del align_model
            gc.collect()
            if device != "cpu":
                torch.cuda.empty_cache()
        except Exception as e:
            _progress(0.50, "※ タイムスタンプ精度向上をスキップしました（結果に影響はありません）")
            print(f"WARNING: アライメントをスキップ（モデルロード失敗: {e}）",
                  file=sys.stderr, flush=True)

        _progress(0.60, "話者分離中...")

        # --- 3. 話者分離 ---
        # Windows ではシンボリックリンク問題で HuggingFace キャッシュが壊れるため、
        # snapshot_download(local_dir=...) でメインモデルをコピーダウンロードし、
        # pyannote の Pipeline を直接ローカルの config.yaml からロードする。
        # サブモデル (segmentation-3.0 等) も cache_dir でアプリローカルに誘導する。
        import pandas as pd
        from huggingface_hub import snapshot_download
        from pyannote.audio import Pipeline as PyannotePipeline

        _diarize_model_name = "pyannote/speaker-diarization-3.1"
        _local_model_dir = os.path.join(_APP_DIR, "models", "speaker-diarization-3.1")
        _progress(0.61, "話者分離モデルを準備中...")
        snapshot_download(
            _diarize_model_name,
            token=args.hf_token,
            local_dir=_local_model_dir,
        )
        _config_path = os.path.join(_local_model_dir, "config.yaml")
        # cache_dir を明示的に渡してサブモデルのダウンロード先を制御
        # （デフォルトの ~/.cache/torch/pyannote/ を使わせない）
        diarize_pipeline = PyannotePipeline.from_pretrained(
            _config_path,
            cache_dir=_APP_HF_CACHE,
        ).to(torch.device(device))

        diarize_kwargs = {}
        if args.num_speakers > 0:
            diarize_kwargs["num_speakers"] = int(args.num_speakers)

        waveform = torch.from_numpy(audio).unsqueeze(0).float()
        diarization = diarize_pipeline(
            {"waveform": waveform, "sample_rate": 16000},
            **diarize_kwargs,
        )
        diarize_segments = pd.DataFrame(
            diarization.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarize_segments["start"] = diarize_segments["segment"].apply(lambda x: x.start)
        diarize_segments["end"] = diarize_segments["segment"].apply(lambda x: x.end)

        _progress(0.85, "話者を割り当て中...")

        # --- 4. 話者割当 ---
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # --- 5. 結果整形 ---
        output = []
        for seg in result["segments"]:
            output.append({
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": seg.get("text", "").strip(),
                "speaker": seg.get("speaker", "不明"),
            })

        _progress(0.95, "完了")
        print(json.dumps(output, ensure_ascii=False))

    except BaseException as e:
        _error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        # main() 外（トップレベル import 等）で落ちた場合の最終防壁
        msg = str(e).replace("\r", "").replace("\n", " | ")
        print(f"ERROR:{msg}", file=sys.stderr, flush=True)
        sys.exit(1)
