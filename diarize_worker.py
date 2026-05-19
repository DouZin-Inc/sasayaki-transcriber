"""話者分離サブプロセスワーカー

transcriber.py から subprocess で起動される。
メインプロセスの CTranslate2 と PyTorch の CUDA コンテキスト競合を回避するため、
話者分離を独立プロセスで実行する。

引数:
    --wav-path     入力WAVファイルパス
    --num-speakers 話者数
    --hf-token     HuggingFace APIトークン
    --gpu-device   GPUデバイス指定 (auto / cpu / cuda:0 など)

出力:
    stdout: JSON配列 [{"start": float, "end": float, "speaker": str}, ...]
    stderr: PROGRESS:<value>:<message> 形式の進捗通知 / ERROR:<message> 形式のエラー
"""

import argparse
import inspect
import io
import json
import os
import sys
import time

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

# Windows: シンボリックリンク作成失敗時にファイルコピーにフォールバック
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


def _progress(value, message):
    """進捗を stderr に出力する。"""
    print(f"PROGRESS:{value:.2f}:{message}", file=sys.stderr, flush=True)


def _error(message):
    """エラーを stderr に出力する。"""
    print(f"ERROR:{message}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description="話者分離ワーカー")
    parser.add_argument("--wav-path", required=True, help="入力WAVファイルパス")
    parser.add_argument("--num-speakers", type=int, default=0, help="話者数（0で自動推定）")
    parser.add_argument("--hf-token", required=True, help="HuggingFace APIトークン")
    parser.add_argument("--gpu-device", default="auto", help="GPUデバイス (auto/cpu/cuda:0)")
    parser.add_argument("--model", default="pyannote/speaker-diarization-community-1",
                        help="話者分離モデル名")
    args = parser.parse_args()

    try:
        _progress(0.60, "話者分離モデルを読み込み中...")

        from pyannote.audio import Pipeline

        try:
            # pyannote v4+
            pipeline = Pipeline.from_pretrained(
                args.model,
                token=args.hf_token,
                cache_dir=_APP_HF_CACHE,
            )
        except TypeError:
            # pyannote v3.x
            pipeline = Pipeline.from_pretrained(
                args.model,
                use_auth_token=args.hf_token,
                cache_dir=_APP_HF_CACHE,
            )

        # GPU転送
        if args.gpu_device == "auto":
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
        elif args.gpu_device != "cpu":
            pipeline.to(torch.device(args.gpu_device))

        _progress(0.65, "話者分離を実行中...")

        # 処理ステップごとに進捗表示するフック
        step_progress = {
            "segmentation": (0.70, "セグメンテーション完了"),
            "embeddings": (0.78, "埋め込み計算完了"),
            "clustering": (0.85, "クラスタリング完了"),
        }
        start_time = time.time()

        def hook(step_name, *_args, **_kwargs):
            if step_name in step_progress:
                value, msg = step_progress[step_name]
                elapsed = int(time.time() - start_time)
                m, s = divmod(elapsed, 60)
                _progress(value, f"{msg}（{m}分{s:02d}秒経過）")

        diar_kwargs = {"hook": hook}
        if args.num_speakers > 0:
            diar_kwargs["num_speakers"] = args.num_speakers
        diarization = pipeline(args.wav_path, **diar_kwargs)

        # 結果をJSON形式で stdout に出力
        result = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            result.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker,
            })

        _progress(0.88, "話者分離完了")
        print(json.dumps(result, ensure_ascii=False))

    except Exception as e:
        _error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
