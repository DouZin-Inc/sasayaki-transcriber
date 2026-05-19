# Sasayaki（ささやき）

ローカル動画文字起こし＆話者分離ツール

## 概要

動画・音声ファイルやマイク録音から高精度な文字起こしと話者分離を行い、字幕ファイル（SRT）やCSVとして出力します。
Adobe Premiere Pro等の動画編集ソフトにそのままインポートできます。

## 特徴

### 文字起こし・話者分離

- **高精度な文字起こし**: OpenAI Whisper (faster-whisper) ベース
- **話者分離**: pyannote-audio による話者識別
- **WhisperX モード**: 文字起こし・アライメント・話者分離を一括実行（UIからインストール可能）
- **モデル切替**: PCスペックに合わせて tiny / base / small / medium / large-v3 を選択可能
- **モデル事前ダウンロード**: 使用前にモデルを個別ダウンロード、ダウンロード済みモデルのみ選択可能
- **高精度タイムスタンプ**: 単語単位でセグメント境界を補正し、字幕のタイミング精度を向上
- **キーワード辞書**: 人名・固有名詞を登録して認識精度向上（インポート/エクスポート対応）
- **プロンプト編集**: Whisper に渡す指示文をカスタマイズ可能
- **多言語対応**: 日本語 / 英語 / 中国語 / 韓国語 / 自動検出

### 入力・出力

- **複数ファイル一括処理**: 複数の動画・音声ファイルをまとめて文字起こし
- **マイク録音**: ブラウザから直接録音して文字起こし
- **複数出力形式**: SRT / CSV / TXT / JSON
- **話者フィルター**: 選択した話者のみでエクスポート可能

### 編集・後処理

- **編集可能プレビュー**: 結果をその場で編集して再エクスポート
- **セグメント編集**: 行ごとの話者・テキスト編集、マージ（統合）、分割、削除
- **一括置換**: 話者名・テキストの一括置換＆再エクスポート
- **タイムスタンプ表示**: プレビューのタイムスタンプ表示ON/OFF切替
- **使用モデル表示**: 処理に使用した Whisper モデルと話者分離モデルを表示

### 管理・操作

- **履歴管理**: 文字起こし結果を自動保存、いつでも復元・再エクスポート可能
- **GPU選択**: 複数GPU環境でも使用デバイスを切り替え可能
- **キャンセル機能**: 実行中の文字起こしを途中でキャンセル
- **簡単操作**: ブラウザベースのGUI（Gradio）、ドラッグ&ドロップ対応
- **ワンクリック導入**: インストーラーで環境構築、start.bat で起動

## 必要環境

- Windows 10/11
- NVIDIA GPU 推奨（CPU のみでも動作可能、ただし低速）

## インストール

[Releases](../../releases) から最新のインストーラー（`SasayakiSetup.exe`）をダウンロードして実行してください。

1. インストーラーを実行
2. セットアップが自動で Python / ffmpeg / 依存パッケージをインストール
3. デスクトップのショートカットから起動

## 使い方

1. 「ファイル」タブで動画・音声ファイルをドラッグ&ドロップ、または「マイク録音」タブで直接録音
2. モデル・話者数・出力形式を選択
3. 「文字起こし開始」をクリック
4. 完了後、ファイルをダウンロード（結果は履歴に自動保存されます）

## モデル比較

| モデル | VRAM目安 | 精度 | 速度 | おすすめ |
|--------|---------|------|------|---------|
| tiny | 1GB | ★★ | 最速 | テスト・確認用 |
| base | 1GB | ★★★ | 速い | GPU なしの PC |
| small | 2GB | ★★★★ | 普通 | バランス重視 |
| medium | 5GB | ★★★★★ | やや遅い | 高精度 |
| large-v3 | 10GB | ★★★★★+ | 遅い | 最高精度（GPU推奨） |

> 使用前に「モデルダウンロード」セクションから事前にダウンロードしてください。

## HuggingFace トークンの設定

話者分離機能を使うには HuggingFace のアクセストークンが必要です。

1. [HuggingFace](https://huggingface.co/join) でアカウントを作成
2. 以下のモデルページで利用規約に同意（3つとも必要）
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1
3. [トークン設定ページ](https://huggingface.co/settings/tokens) でアクセストークンを作成（Read権限）
4. Sasayaki の画面でトークンを入力（自動保存されるため初回のみ）

> 詳しい手順はアプリ内の「使い方ガイド」をご覧ください。

## アップデート

最新のインストーラーをそのまま実行してください。アプリ本体のみ上書きされ、Python / ffmpeg / 設定はそのまま引き継がれます。

## 技術スタック

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - 高速文字起こし
- [WhisperX](https://github.com/m-bain/whisperX) - 文字起こし・アライメント・話者分離の一括実行
- [pyannote-audio](https://github.com/pyannote/pyannote-audio) - 話者分離
- [Gradio](https://gradio.app/) - Web GUI
- [ffmpeg](https://ffmpeg.org/) - 音声抽出

## 開発者

- **y.nakai** — 企画・設計・実装
  [@DZ-nakai](https://github.com/DZ-nakai) · [株式会社同人 (DouZin Inc.)](https://douzin.co.jp)

本プロダクトは株式会社同人にて単独で企画から実装まで担当しました。
受託開発・技術相談のご連絡は [y.nakai@douzin.co.jp](mailto:y.nakai@douzin.co.jp) または GitHub Issues までお気軽にどうぞ。

## ライセンス

[MIT License](LICENSE) — Copyright (c) 2026 DouZin Inc.
