# Ollo Tools 再現：Gemini ベース作業分析（分類・説明・改善提案）設計ドキュメント

- **作成日**: 2026-06-10
- **ステータス**: 設計レビュー中 / 実装計画待ち
- **対象**: Phase 1 の MVP（CLIP ゼロショット）の上に、Ollo Factory **Tools**（標準化）相当の機能を載せる
- **前提spec**: `2026-06-09-ollo-benchmark-egocentric-work-analysis-design.md`（基盤）／`2026-06-09-frontend-web-ui-design.md`（UI）

---

## 1. 背景と狙い

### 1.1 Phase 1 の到達点
- 共通スキーマ `SegmentList{video_id, fps_sampled, label_vocabulary, segments[], source}`、`Segment{start_sec, end_sec, label, confidence}`
- Track A（Gemini 単発・窓分割）、Track B（CLIP 境界 + CLIP 類似度ラベリング）、評価基盤（F1@{10,25,50}/Edit/Acc）
- Web UI（アップロード→処理→結果：Gantt/テーブル/統計）

### 1.2 今回の狙い：Ollo Factory **Tools** の再現
Tools の3本柱は **(1) 手順書作成 (2) ムダ作業分類 (3) 改善案提案**。これに **ヒント画像** と **ワンショット（語彙の半自動立ち上げ）** が乗る。本specはこれらを、現状 CLIP で粗く行っている**最終ラベリングを Gemini に置換**したうえで実現する。

主要な設計決定（壁打ち確定事項）:
1. **正味/付随/ムダ の3分類は Gemini が映像から自動推定**（カテゴリは固定3値）
2. **役割分担**: CLIP が粗い境界を出し（Stage 1）、**Gemini が境界を見直しつつ意味付け**（Stage 2）
3. **作業内容の説明はセグメント単位（文脈依存）**
4. **改善提案はセグメント単位**（ムダ・付随に付与）
5. **Stage 2 はリッチ単一呼び出し**（窓ごとに Gemini 1回で label+category+description+improvement を一括出力）
6. **ヒント画像は拡張口のみ**（スキーマ・引数・プロンプト差し込み口を用意。描画UI/物体追跡は次フェーズ）
7. **トラックは3本**（標準=Gemini精緻化／CLIPのみ／Gemini単発）を同一評価軸で横並び比較

---

## 2. スコープ

### 2.1 含む
| # | 項目 |
|---|---|
| データモデル | `Segment` に `category`・`description`・`improvement` を追加（後方互換）。`Hint` 追加（拡張口） |
| Stage 2 | 新モジュール `label_gemini`：CLIP境界整列の窓ごとに Gemini 1回で 境界見直し+label+category+description+improvement |
| 入力支援 | PDF作業標準書のパース（テキスト＋画像フォールバック）、語彙の自動提案（ワンショット） |
| 出力 | Gantt（カテゴリ色）／統計（カテゴリ別＋ラベル別集計）／手順書ドラフト更新／改善提案表示 |
| トラック | 標準（Gemini・既定）／CLIPのみ／Gemini単発 の3本 |
| 再現性 | temperature=0、model id・プロンプト版の記録、生 Gemini 応答の永続化 |
| UI | Gantt 反映バグ修正、3トラック選択、enrich欠損時の優雅な劣化 |

### 2.2 含まない（スコープ外）
| 除外項目 | 扱い |
|---|---|
| ヒント画像の描画UI・物体追跡 | 次フェーズ。スキーマ・引数・プロンプト差し込み口のみ用意 |
| category/description/improvement の定量評価 | 正解注釈が無いため今回は **UI enrichment として非評価**。評価は label/境界のみ（従来の F1/Edit/Acc） |
| 再分析の Stage1 キャッシュ | 今回は注記のみ。ヒントUI実装フェーズで効率化 |
| 外部API（Gemini）へ現場映像を送るデータガバナンス | リスク注記のみ。オンプレ要件は別途 |
| 改善提案の集計・パターン化 | 今回はセグメント単位。表示側で重複を畳む（§8.4） |
| 作業抜け・順番違い検知（手順チェック） | **Ollo Factory Training の機能**であり Tools 再現のスコープ外。将来 `check_procedure(seg_list, expected_steps) -> list[Deviation]`（順序付き標準手順とのシーケンス整列）として Stage 3 に追加可能（§9.2 の精度限界に留意） |

---

## 3. データモデル

### 3.1 `Segment` 拡張（後方互換）
```python
from typing import Optional

@dataclass
class Segment:
    start_sec: float
    end_sec: float
    label: str
    confidence: float = 1.0
    category: Optional[str] = None       # "seimi" | "fuzui" | "muda"（未分類は None）
    description: Optional[str] = None     # 文脈依存の作業内容説明
    improvement: Optional[str] = None     # 改善提案（主に muda/fuzui に付与、無ければ None）
```
- 新フィールドはすべて `Optional` 既定 `None`。`from_json` は `Segment(**seg)` のままで、**旧JSON（新キー無し）も読める**。`to_json`（asdict）は新キーを `null` で出力する。
- **評価は start_sec/end_sec/label のみ使用**するため、追加フィールドは評価に無影響。

### 3.2 カテゴリの正規キー（固定3値）
| 正規キー | 表示名 | 色（UI） | 定義（IE 作業分析） |
|---|---|---|---|
| `seimi` | 正味作業 | 緑 `#22C55E` | 付加価値を生む主作業（例：組立・締結・加工そのもの） |
| `fuzui` | 付随作業 | 橙 `#F97316` | 主作業に必要だが価値を生まない補助（例：部品を取る・工具を持つ・段取り） |
| `muda`  | ムダ作業 | 青 `#3B82F6` | 価値を生まない非作業（例：手待ち・探す・手戻り・無駄歩行） |

- 表示名・色はUI側マッピング（モックアップ `ui-mockup.html` と一致）。
- **カテゴリはラベルの属性ではなく区間の属性**（同じ「歩行」でも文脈で付随/ムダが変わる）。Gemini が区間ごとに推定する。

### 3.3 `Hint`（拡張口・今回はUI未実装）
```python
@dataclass
class Hint:
    label: str                                  # ユーザーが付ける物体/作業名
    frame_sec: float                            # 対象フレーム時刻
    bbox: Optional[tuple] = None                # 正規化 (x, y, w, h)。None = フレーム全体
    note: Optional[str] = None
```
- パイプラインは `hints: list[Hint] | None` を受け取り、**今回は hint.label/note を Gemini プロンプトに「既知の物体・着目点」として差し込むのみ**。
- bbox 切り出し画像の投入・描画UI・物体追跡は次フェーズ。

---

## 4. アーキテクチャ

### 4.1 全体パイプライン
```
動画（〜20分超・非反復）  ＋ 任意PDF（作業標準書）  ＋ 任意ヒント
  ↓
[Stage 0] フレーム抽出・顔ブラー        … 既存（ingest）
  ↓
[Stage 1] CLIP埋め込み → ruptures で粗い境界候補   … 既存（embed / presegment）
  ↓
[Stage 2] ★Gemini 精緻化（新規 label_gemini）
   ・CLIP境界に整列した窓ごとに Gemini を1回呼ぶ
   ・窓内で 境界見直し ＋ label ＋ category ＋ description ＋ improvement を出力
   ・PDF参照文脈・ヒントをプロンプトに注入
  ↓
[Stage 3] 出力整形（report / aggregate）
   ・タイムライン不変条件の強制（連続・非重複・全長被覆）
   ・Gantt / 統計（カテゴリ別＋ラベル別）/ 手順書ドラフト
```

### 4.2 Stage 2：`src/pipeline/label_gemini.py`
```python
def label_gemini(
    video_path: str,
    label_vocabulary: list[str],
    boundary_timestamps: list[float],
    *,
    blur_faces: bool = False,
    hints: list[Hint] | None = None,
    reference_context: str | None = None,   # PDF由来の参照文脈（§6.1）
    model: str = "gemini-2.5-pro",
    source: str = "track_std",
) -> SegmentList
```

処理手順:
1. **粗区間の構築**: `boundary_timestamps` から `[(start, end), ...]` を作る（既存 `label_zeroshot` と同一ロジック）。
2. **窓へのグルーピング（境界整列・H3/A）**: 窓は**粗区間の整数個**をまとめたもの。窓の長さ目安は ~5分。**粗区間の途中で窓を切らない**。隣接窓は**1粗区間だけオーバーラップ**させ文脈を渡す。
3. **フレーム予算（H）**: 窓あたり最大フレーム数（既定 `MAX_FRAMES_PER_WINDOW = 30`）、粗区間あたりサンプル数（既定 `FRAMES_PER_SEGMENT = 3`）を上限とし、超える場合は等間隔間引き。
4. **Gemini 1回呼び出し（案1）**: プロンプトに〈ラベル語彙＋カテゴリ定義（§3.2）＋境界見直し指示＋出力スキーマ＋reference_context＋hints〉を入れ、`{start_sec, end_sec, label, category, description, improvement, confidence}` のJSON配列を要求。`temperature=0`。
5. **正規化**:
   - **category 正規化（F）**: 「正味/value-adding」→`seimi` 等の表記揺れを3キーへ写像。未知値は `None` にしてログ。
   - **語彙外ラベル（H2/G）**: 「手待ち」「その他」など語彙に無いラベルの出力を許可。出力 `SegmentList.label_vocabulary` に**実効的に追記**する。
   - **語彙外ラベルの名寄せ（J）**: 実効語彙へ追記する**直前**に簡易シノニム正規化を通す。①trim・文字種統一 → ②ユーザー入力語彙と完全一致すればそれを採用 → ③頻出表記揺れマッピング辞書（定数テーブル `_LABEL_SYNONYMS`：例「手待ち時間/待機 → 手待ち」「探している → モノ探し」）に合致すれば既定表現へ強制置換。これにより `aggregate`（§7.1）の集計が同一作業で分裂するのを防ぐ。
   - 時刻を窓内にクランプ。
6. **stitch と不変条件（A）**: 全窓の結果を結合。**オーバーラップ領域は窓のコア（非重なり）に中心がある区間を採用**し重複排除。最後に `sort → 隙間埋め → 重複クリップ` を施し、**[0, total_duration] を隙間なく被覆する連続・非重複列**を保証する。隣接同一 (label, category) はマージ。
   - **enrich マージポリシー（K・コア優先型）**: 隣接同一 (label, category) をマージする際、`description`/`improvement` は**コンカチせず、窓のコア領域（非オーバーラップ部）に属する側のテキストを正として採用**し、他方は破棄する。窓Aの末尾と窓Bの先頭が同一作業のときに表記揺れ・文脈断絶した2文が並ぶことを防ぐ。同一窓内のマージ（オーバーラップ無関係）では時間が長い側を採用。純関数として stitch ロジック内に実装しテスト対象とする。
7. **再現性（H4/F）**: 各窓の**生 Gemini 応答**・model id・プロンプト版ハッシュを `results/{video_id}_{source}_raw.json` に保存。temperature=0 は best-effort 決定性であり完全な再現は API 上保証されないため、出力の永続化で監査・リプレイを担保する。

### 4.3 トラック構成（3本横並び）
| source キー | 構成 | 位置づけ | enrich(category/desc/improvement) |
|---|---|---|---|
| `track_std`（**既定**） | CLIP境界 + Gemini精緻化（`label_gemini`） | 本番パイプライン | ✅ あり |
| `track_b` | CLIP境界 + CLIP類似度（`label_zeroshot`） | API不要・高速・比較用 | なし（None） |
| `track_a` | Gemini単発（`label_vlm_single`） | ベースライン比較 | ✅ あり（同プロンプト規約で付与） |

- enrich を `track_std` と `track_a` の両方が出力し、UI はトラック非依存に動く。`track_b` は enrich を持たず、**UIは欠損時に優雅に劣化**（§8.5）。

### 4.4 トラックレジストリ（N・将来のモデル差し替え口）
- `jobs` の track 分岐は if/elif ではなく**レジストリ辞書**にする。差し替え単位は「**動画＋語彙 → SegmentList**」のトラック全体（TASOT 等の時系列アクションセグメンテーションモデルは境界検出と分類を同時に行うため、Stage 2 単体ではなく Stage 1+2 をまとめて置き換えうる）。

```python
# src/web/jobs.py
class TrackRunner(Protocol):
    def __call__(self, video_path: str, label_vocabulary: list[str],
                 **opts) -> SegmentList: ...

TRACK_RUNNERS: dict[str, TrackRunner] = {
    "std": run_track_std,   # CLIP境界 + Gemini精緻化（既定）
    "b":   run_track_b,     # CLIP境界 + CLIP類似度
    "a":   run_track_a,     # Gemini単発
    # 将来例: "tasot": run_track_tasot  ← 関数1つ＋この1行で追加
}
```

- `SegmentList` スキーマ・評価スクリプト・UI はすべて**トラック非依存**のため、新トラックは TrackRunner 実装＋レジストリ登録のみで評価・表示まで自動的に乗る。enrich を持たないトラックは `track_b` と同じ劣化パス（§8.5）に乗る。
- track 既定値を `std` にする。

---

## 5. ワンショット：語彙の自動提案（F1）

`src/pipeline/propose_labels.py`
```python
def propose_labels(
    video_path: str,
    *,
    reference_context: str | None = None,
    blur_faces: bool = False,
    model: str = "gemini-2.5-pro",
    max_labels: int = 12,
) -> list[str]
```
- 動画から等間隔サンプリングしたフレーム（＋任意PDF参照文脈）を Gemini に渡し、要素作業ラベル候補を提案。
- **発火と非ブロッキング（C）**: アップロード完了後（PDFがあればそのパース後）に**非同期**で実行し、結果をラベルフォームへプリフィル。**ユーザーは提案を待たずに手入力でき**、提案失敗時は素通り（フォームは空のまま手入力可）。
- 提案語彙はユーザーが編集して確定。確定語彙が `analyze` に渡る。

---

## 6. 入力支援：PDF作業標準書（H1）

`src/pipeline/parse_reference.py`
```python
def parse_reference(pdf_path: str, *, model: str = "gemini-2.5-pro") -> str
    # 戻り値: reference_context（Gemini プロンプトに注入する参照文脈テキスト）
```
- **テキスト抽出を試行**（pdfplumber 等）。
- **スキャン画像PDFフォールバック（B）**: 抽出テキストが乏しい場合、**PDFページを画像として Gemini に渡し**、作業手順・着目点・想定カテゴリを要約して `reference_context` を得る。
- **リソースガードレール（L）**: 数十〜100ページ超の標準書でコンテキスト溢れ・コスト超過・ハングを防ぐ。
  - `MAX_PDF_IMAGE_PAGES = 10`：画像フォールバックの上限ページ数。超過時は**先頭ページ優先**で切り出し（手順書は前半に主手順が集中する想定）、切り捨てた旨を `reference_context` 冒頭に注記。
  - 画像化は低DPI（目安 72〜100）＋ Pillow リサイズ・圧縮で入力トークンを最小化。
  - PDFパース処理全体を `asyncio.wait_for`（目安 60秒）でタイムアウトラップし、失敗時は `reference_context = None` で素通り（分析はPDFなしで続行）。
- `reference_context` は**自由文の参照**として `label_gemini` と `propose_labels` のプロンプトに注入する（ラベルへの厳密binding はしない＝ユーザーがラベルを改名しても破綻しない）。

---

## 7. 出力（Stage 3）

### 7.1 集計（F3）：`src/pipeline/aggregate.py`
```python
def aggregate(seg_list: SegmentList) -> dict
    # by_category: {seimi/fuzui/muda: {total_sec, count, ratio}}
    # by_label:    {label: {total_sec, count, mean_sec}}
    # total_sec
```
- **カテゴリ別**（円グラフ用）と **ラベル別 合計/平均/回数**（標準時間分析の基礎）を返す。

### 7.2 手順書ドラフト更新（H5）：`report.to_procedure_markdown`
- 各 Step に **description**・**category（表示名）**・所要時間を出力。末尾に **ラベル別集計（F3）** とカテゴリ別内訳を付す。
- 改善提案（improvement）がある Step には「改善ヒント」を併記。

---

## 8. API / UI

### 8.1 ルート
| メソッド/パス | 変更 | 内容 |
|---|---|---|
| `POST /upload` | 変更 | 動画 ＋ **任意PDF** を受領・保存。PDFは `parse_reference` 対象 |
| `POST /propose-labels`（新） | 追加 | job_id（＋PDF）から **語彙候補を提案**（非同期、§5）。フォームにプリフィル |
| `POST /analyze` | 変更 | `track ∈ {std, b, a}`、任意 `hints`(JSON)、`reference_context` を使用 |
| `GET /status/{job_id}` | 既存 | ポーリング（変更なし） |
| `GET /results/{job_id}?track=` | 変更 | enrich込みセグメントを返す。3トラック切替 |
| `GET /video/{job_id}` | 既存 | Range ストリーミング（変更なし） |
- **再分析フック**: `/analyze` を `hints` 付きで再呼び出し可能にする（拡張口。今回はAPI受け口のみ）。

### 8.2 Gantt 反映バグの修正（真因）
現状、結果ロード時に `_timeline.html` がテーブルを描くだけで、クライアントの Gantt/統計へ `segments-loaded` が飛んでいない。
→ **結果ロード時に enrich 済みセグメント（category 込み）を `segments-loaded` で発火**し、Gantt/統計をクライアント描画する。

### 8.3 Gantt の配色
- **カテゴリ色（正味=緑/付随=橙/ムダ=青）で塗る**（ラベルハッシュ色ではなく）。ラベル文字は行ラベル・ツールチップで保持し識別性を担保。

### 8.4 説明・改善提案の表示（F2 はセグメント単位）
- セグメント詳細／サイドバー／ツールチップに **description**。
- ムダ・付随区間に **improvement**。**同一内容が並ぶ場合は表示側で畳む**（重複抑制）。

### 8.5 enrich 欠損時の優雅な劣化（D）
- `track_b`（CLIPのみ）等で category/description/improvement が `None` の場合、UIは **ラベルハッシュ色へフォールバック**し、説明・改善欄を非表示にする。

### 8.6 ラベルフォーム
- 提案語彙（F1）をプリフィル。PDFドロップを `/upload` に接続。

---

## 9. 評価・ベンチマーク

### 9.1 指標と対象
- 評価指標は従来通り **F1@{10,25,50}/Edit/Acc**、対象は **label/境界のみ**。3トラック（`track_std`/`track_b`/`track_a`）を同一スクリプトで横並び比較。
- **category/description/improvement は今回非評価**（正解注釈が無い）。将来カテゴリ注釈を追加したら category Accuracy を別途定義（拡張口）。
- 新既定 `track_std` が CLIP（`track_b`）や単発（`track_a`）に対し境界/ラベルで優位かは**実測で確認**（無条件に優位と仮定しない）。

### 9.2 「論理的境界」と正解データの乖離（M）
- `track_std` の Gemini 境界補正は IE 的な論理（「工具に手が触れた瞬間を開始とする」等）で境界を動かすため、アノテーターの物理タイムスタンプと 1〜2 秒ズレ、**定性的には正しいのに F1 が `track_b` より悪化して見える逆転現象**が起こりうる。
- 対応:
  1. 評価スクリプトに**境界ズレ分布の詳細ログ**（閾値 10/25/50 のどこで落ちているかのブレークダウン）を追加し、ズレ幅由来の悪化を切り分け可能にする。
  2. 実装コメント・テストドキュメントに「数値の絶対的優劣だけでなく、過分割の抑制・IE 視点の論理的正しさを**人間がサンプル目視検証する**ステップが必要」と明記する。

### 9.3 精度の構造的限界（注記）
- **幻覚リスク**: description にフレームに写っていない内容が混入しうる。プロンプトで「観察できた事実のみ・推測は明示」を強制し、confidence を UI に表示して低信頼区間を視覚的に区別する。
- **短時間作業の欠落**: サンプリング fps と粗区間あたりフレーム予算の制約上、数秒以下の作業はフレームに写らず欠落しうる。「**検知されなかった ≠ 行われなかった**」であり、本システムの出力から作業の不実施を断定してはならない（作業抜け検知は Training 機能・スコープ外、§2.2）。

---

## 10. コンポーネント境界（単一責務）

| コンポーネント | 責務 | 依存 | 状態 |
|---|---|---|---|
| `ingest` / `embed` / `presegment` | Stage0/1（フレーム・埋め込み・境界） | OpenCV, CLIP, ruptures | 既存・無変更 |
| `label_zeroshot` | Stage2（CLIP・track_b） | CLIP | 既存・無変更 |
| `label_vlm_single` | Track A（Gemini単発） | Gemini | enrich対応に小改修 |
| `label_gemini`（新） | Stage2（Gemini精緻化・track_std） | Gemini | 新規 |
| `propose_labels`（新） | 語彙の自動提案（F1） | Gemini | 新規 |
| `parse_reference`（新） | PDF→参照文脈（H1/B） | pdfplumber, Gemini | 新規 |
| `aggregate`（新） | カテゴリ別/ラベル別集計（F3） | — | 新規 |
| `report` | タイムライン/手順書ドラフト | — | H5で更新 |
| `evaluate` | 指標算出 | — | 3トラック対応のみ |
| `web/routes` `web/jobs` | API・ジョブ管理 | 上記 | propose-labels/PDF/3トラックで更新 |

`label_zeroshot`・`label_vlm_single`・`label_gemini` は同一インターフェース（入力＝映像＋語彙＋境界、出力＝`SegmentList`）を保ち差し替え可能（元spec原則の踏襲）。さらに上位の差し替え単位として `TRACK_RUNNERS` レジストリ（§4.4）があり、TASOT 等の境界検出一体型モデルもトラック1本として追加できる。

---

## 11. テスト計画

| 対象 | 内容 | 外部依存 |
|---|---|---|
| schema | 新フィールド round-trip、旧JSON読み込み後方互換 | なし |
| `label_gemini` | JSONパース、category正規化、語彙外ラベル追記＋**名寄せ（J）**、時刻クランプ | Gemini をモック |
| 窓・stitch | **境界整列**（粗区間を割らない）、オーバーラップ採用ルール、**不変条件**（連続・非重複・全長被覆）、**enrichマージポリシー（K・コア優先）** | なし（純関数） |
| `propose_labels` | フレームサンプリング、提案パース、失敗時の空リスト | Gemini をモック |
| `parse_reference` | テキスト抽出、画像フォールバック分岐、**ページ上限・タイムアウト・失敗時None素通り（L）** | pdf/Gemini をモック |
| `aggregate` | カテゴリ別/ラベル別の合計・平均・回数 | なし |
| `report` | description/category/improvement/集計を含む手順書 | なし |
| `web/routes` | propose-labels、PDF付upload、3トラックanalyze、enrich込み results | Gemini をモック |

- Gemini/PDF 呼び出しはすべてモックし、CIはAPIキー無しで全件パス。

---

## 12. リスクと留意点

| リスク | 対応 |
|---|---|
| Gemini が境界を動かしすぎ/過分割 | プロンプトでCLIP境界を初期値として提示、信頼度低時は維持。不変条件で整形 |
| 窓またぎの境界不整合 | 1区間オーバーラップ＋コア採用ルール＋後処理整形（§4.2-6） |
| スキャンPDFでテキスト抽出失敗 | 画像フォールバックで Gemini に直接読ませる（§6） |
| Gemini 非決定性 | temperature=0 ＋ 生応答永続化（best-effort と明記） |
| 語彙外ラベルで評価が悪化して見える | 正しい挙動（正解に無ければ誤り）。実効語彙に追記し記録 |
| 語彙外ラベルの表記揺れで集計が分裂 | 名寄せレイヤー（§4.2-5 J）：シノニム辞書で既定表現へ正規化 |
| 窓またぎマージで description が断絶・競合 | コア優先マージポリシー（§4.2-6 K） |
| 大部数スキャンPDFでコスト・ハング | ページ上限・低DPI圧縮・タイムアウト（§6 L） |
| Gemini境界補正でF1が見かけ悪化 | ズレ分布の詳細ログ＋目視検証ステップの明記（§9.2 M） |
| 現場映像の外部送信 | データガバナンスは今回スコープ外・注記のみ。顔ブラーは送信前に適用 |
| コスト（窓×フレーム） | フレーム予算上限を定数化、窓は尺に比例せず粗区間数に比例 |

---

## 13. 実装順序（参考・writing-plans で詳細化）

1. スキーマ拡張（Segment/Hint）＋round-trip テスト
2. `label_gemini`（窓整列・stitch・不変条件・正規化）＋テスト（Geminiモック）
3. `aggregate`＋`report` 更新＋テスト
4. `jobs`/`routes` に `track_std`・3トラック・enrich results を統合＋テスト
5. `parse_reference`（PDF・画像フォールバック）＋`propose_labels`＋`/upload`PDF・`/propose-labels`＋テスト
6. UI：Gantt反映バグ修正・カテゴリ配色・description/improvement表示・優雅な劣化・語彙プリフィル
7. 評価スクリプトを3トラック対応に更新
8. 統合テスト（Geminiモックでパイプライン一気通し）

---

## 付録：壁打ち確定事項の一覧
- 分類: Gemini自動推定 / 3値固定（seimi/fuzui/muda）/ 区間属性
- 役割: CLIP境界 → Gemini が境界見直し＋意味付け
- 説明: セグメント単位（文脈依存）
- 改善提案: セグメント単位
- Stage2: リッチ単一呼び出し（窓ごとGemini1回）
- ヒント画像: 拡張口のみ（描画UI/追跡は次）
- トラック: 3本（標準=Gemini既定 / CLIPのみ / Gemini単発）
- 追加機能: PDF活用(H1) / 語彙自動提案(F1) / 改善提案(F2) / ラベル別集計(F3) すべて今回
- 設計の穴: H1〜H5 ＋ 残課題A〜I をすべて折り込み
- 外部レビュー反映（J〜N）: J=語彙外ラベル名寄せ / K=enrichコア優先マージ / L=PDFガードレール / M=評価ズレ分布ログ＋目視検証注記 / N=TRACK_RUNNERSレジストリ（TASOT等の差し替え口）
- スコープ外追加確認: 作業抜け・順番違い検知は Ollo Factory **Training** 機能のため見送り（§2.2 に将来案 `check_procedure` を注記）
