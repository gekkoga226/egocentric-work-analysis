# Webフロントエンド設計ドキュメント（FastAPI + htmx + Alpine.js）

- **作成日**: 2026-06-09
- **ステータス**: 設計承認済み / 実装計画待ち
- **対象**: egocentric-work-analysis のWeb UI。既存のCLIパイプライン（Track A/B・評価基盤）の上に載せる、動画照合可能な作業分析フロントエンド
- **関連**:
  - 親スペック: [2026-06-09-ollo-benchmark-egocentric-work-analysis-design.md](2026-06-09-ollo-benchmark-egocentric-work-analysis-design.md)
  - バックエンド実装計画: [2026-06-09-zero-shot-action-segmentation.md](../plans/2026-06-09-zero-shot-action-segmentation.md)

---

## 1. 目的と位置づけ

親スペックで「UIフロントエンドは後続フェーズ」としていた部分を、**独立フェーズ**として設計する。バックエンドのCLIパイプライン（`src/pipeline`・`src/evaluate`）は**変更せず**、その上に薄いWeb層（`src/web/`）を載せる。

### 1.1 中心的ワークフロー

**アップロード → 分析 → 結果確認 → 動画照合** の一連を1画面で行う（ユーザー回答: D）。
- 「分析して結果を見る」(A) と「結果を動画と突き合わせて検証する」(B) の両方を満たす

### 1.2 運用環境の前提

- 当面は**ローカル / 社内LANのシングルユーザー**運用。認証なし（ユーザー回答: C）
- 将来のチーム共有（マルチユーザー）を見据え、**差し替えの継ぎ目**を設計に残す
- **オンプレ / 閉域網での動作を保証**（後述のベンダリング）

### 1.3 動画確認の操作モデル

**双方向**（ユーザー回答: C）：
- タイムラインの行クリック → 動画がその時刻にシーク（主）
- 動画再生中 → 現在位置に対応するタイムライン行をハイライト（副）

---

## 2. 技術選定

**FastAPI + htmx + Alpine.js**（Approach 2）。

| 技術 | 役割 |
|---|---|
| FastAPI | 既存パイプラインを呼ぶHTTP層。テンプレート描画・動画配信 |
| htmx | サーバー連携：アップロード・分析トリガ・進捗ポーリング・結果断片の差し込み |
| Alpine.js | クライアント側リアクティブ：動画↔タイムライン同期、UI状態管理 |
| Jinja2 | サーバーサイドテンプレート（HTML断片の生成） |

**役割分担の原則**：htmx は「サーバーとやり取りする操作」、Alpine.js は「ページ内で完結する状態同期（`video.currentTime` 監視など）」。動画位置イベントは毎秒発火するため、サーバー往復を伴うhtmxではなくAlpine.jsで処理する。

### 2.1 オンプレ対応：ライブラリのベンダリング

htmx・Alpine.js・CSSは**CDNを使わず `src/web/static/vendor/` に同梱**する。閉域網の工場でもCDN取得失敗で起動不能にならないようにする。CSSはTailwindのCDN版を使わず、ビルド済みCSSの同梱または素のCSSに留める。

---

## 3. アーキテクチャ

```
ブラウザ
  ├── htmx        → アップロード / 分析トリガ / ポーリング / 結果差し込み
  ├── Alpine.js   → 動画↔タイムライン同期、UI状態
  └── HTML5 video → /video/{job_id} から 206 Range ストリーム

FastAPI（src/web/ — 既存パイプラインの薄いラッパー）
  ├── GET  /                  メイン画面
  ├── POST /upload            大容量動画をチャンクでディスク保存（メモリに載せない）
  ├── POST /analyze           ジョブ登録 → 別スレッド（max_workers=1）で実行
  │                            中で既存 label_zeroshot / label_vlm_single を呼ぶ
  ├── GET  /status/{job_id}   インメモリ job dict を返す（即応）
  ├── GET  /results/{job_id}  results/*.json を読んで タイムラインHTML を返す
  └── GET  /video/{job_id}    Range対応(206)で動画ストリーム配信

状態管理:
  ・進行中  = インメモリ dict {job_id: {status, stage, track, error}}  ← 再起動で消える(将来の継ぎ目)
  ・完了結果 = 既存 results/{video_id}_{source}.json をそのまま利用（二重管理しない）
  ・job_id  = 一意化した video_id（§5.2参照）
```

### 3.1 実行モデル（イベントループを塞がない）

Track B の CLIP 推論（20分＝1200フレーム）や Track A の Gemini 複数回呼び出しは**数分**かかる。これを `async def` 内で直接回すとイベントループを占有し、`/status` ポーリングが返らずUIが固まる。

- 重い処理は **`run_in_executor`（`ThreadPoolExecutor(max_workers=1)`）** に逃がす
- `max_workers=1` で**重いジョブを直列化**し、同時実行によるOOM・スラッシングを防ぐ
- async エンドポイント（`/status` 等）は常に即応する
- **CLIPモデルはアプリ起動時にプリウォーム**（初回 `/analyze` でのモデルDL待ちで「固まった」ように見えるのを防ぐ）
- マルチユーザー化（将来）では、このインメモリレジストリ＋スレッド実行を**永続ジョブストア＋タスクキューに差し替える**継ぎ目とする

---

## 4. 画面レイアウトとインタラクション

### 4.1 3ペインのシングルページ

URLは遷移せず、htmxがコンテンツ断片を差し替える。

```
┌─────────────────────────────────────────────────┐
│  ヘッダー（タイトル + 処理状態インジケーター）   │
├──────────┬──────────────────┬───────────────────┤
│  サイド  │   動画プレイヤー  │  セグメント       │
│  パネル  │   （中央・大）   │  タイムライン     │
│          │                  │  （右・スクロール）│
│ ・アップ │  ▶ [====|====]   │  ┌──────────────┐│
│  ロード  │                  │  │ 00:00 作業A  ││
│ ・ラベル │                  │  │ 00:12 作業B  ││
│  入力   │                  │  │ 00:31 作業C  ││
│ ・Track │                  │  └──────────────┘│
│  選択   │                  │  Track A / B 切替 │
│ ・実行  │                  │  評価指標テーブル  │
└──────────┴──────────────────┴───────────────────┘
```

### 4.2 動画↔タイムライン同期（Alpine.js・サーバー往復なし）

- タイムライン行クリック → `video.currentTime = segment.start_sec`
- 動画 `@timeupdate` → 現在時刻に対応する行に `ring-2` ハイライト
- Track A/B 切替タブ → htmx が `/results/{job_id}?track=a|b` を叩きタイムラインペインを差し替え

---

## 5. ファイル構成とコンポーネント

既存の `src/pipeline`・`src/evaluate` には手を入れず、`src/web/` を新設する。

```
src/web/
├── __init__.py
├── app.py            # FastAPI生成・ルーター登録・静的マウント・起動時プリウォーム
├── routes.py         # 各エンドポイント（HTTP入出力とテンプレート描画のみ）
├── jobs.py           # JobRegistry + run_in_executor 実行管理（分析ロジックは持たない）
├── video_stream.py   # Range(206)対応の動画ストリーム配信
├── ids.py            # video_id の一意化・サニタイズ（§5.2, §5.3）
├── templates/
│   ├── index.html        # 3ペインのメイン画面（Alpine.js状態のルート）
│   ├── _label_form.html  # アップロード後に返すラベル入力フォーム断片
│   ├── _timeline.html    # セグメント一覧 + 評価指標テーブルの断片
│   ├── _status_running.html  # 進捗（ポーリング継続）断片
│   └── _status_done.html     # 完了（/resultsを1回呼ぶ）断片
└── static/
    ├── app.css
    └── vendor/           # htmx.min.js, alpine.min.js（CDN不使用・同梱）
```

| ファイル | 責務 | 依存 |
|---|---|---|
| `app.py` | アプリ組み立て・起動時CLIPプリウォーム | routes |
| `routes.py` | HTTP入出力・テンプレート描画 | jobs, video_stream, ids, 既存pipeline/evaluate |
| `jobs.py` | ジョブ登録・状態遷移・別スレッド実行 | 既存 label_zeroshot / label_vlm_single |
| `video_stream.py` | Rangeヘッダ解析→206応答 | — |
| `ids.py` | video_id 一意化・登録済みID検証 | — |

`jobs.py` は既存パイプライン関数を呼ぶだけで、**分析ロジックを一切持たない**。

### 5.2 video_id の一意化（同名ファイル衝突対策）

「video.mp4」を複数回アップすると video_id が衝突し、結果JSONとジョブ状態を破壊する。保存時に `{サニタイズ済みファイル名}_{短いハッシュまたはタイムスタンプ}` で一意化し、それを video_id として既存パイプラインに渡す。IDは1つのまま一意性を担保。

### 5.3 パストラバーサル対策

`/video/{job_id}`・`/results/{job_id}` は `job_id` を**必ずジョブレジストリの存在チェックを通してから**ファイルにアクセスする（登録済みIDのみ受理）。ユーザー入力を直接パス結合しない。

---

## 6. データフロー

```
1. 動画選択 → POST /upload (multipart, チャンクでディスク保存)
     → ids で video_id 一意化・登録
     ← htmx が _label_form.html を #sidebar に差し込む

2. ラベル入力 + Track選択 + 実行 → POST /analyze {video_id, labels, track}
     → GEMINI_API_KEY チェック（Track A時、未設定なら案内断片を返す）
     → jobs.register(video_id) → run_in_executor(max_workers=1) で分析開始
     ← htmx が _status_running.html（hx-trigger="every 2s" で /status をポーリング）を返す

3. /status/{job_id} ポーリング
     ← running の間は _status_running.html を返す
     ← done になったら _status_done.html を返す
        （= ポーリング要素を持たず、hx-trigger="load" で /results を1回だけ叩く断片）
        → これでポーリングが確実に停止する

4. GET /results/{job_id}?track=b
     → results/{video_id}_track_b.json を読み込み
     → annotations/{video_id}.json が存在すれば compare_systems で評価指標を算出
     ← _timeline.html（セグメント行 + 評価指標テーブル）を #timeline に差し込む

5. 動画↔タイムライン同期（クライアント側・Alpine.js のみ）
     - 行クリック → video.currentTime = segment.start_sec
     - video @timeupdate → 現在時刻に対応する行をハイライト
```

### 6.1 評価指標テーブルの正解紐付け規約

`annotations/{video_id}.json` という命名規約で正解を対応付ける。存在すれば `compare_systems` を呼んで F1@{10,25,50} / Edit / Acc を表示、無ければ非表示。

---

## 7. エラー処理

| 事象 | 扱い |
|---|---|
| 非対応の動画形式 | /upload で拡張子チェック、または analyze 時に cv2 が開けなければ status=error |
| 分析中の例外（Gemini API失敗等） | job を status="error", error メッセージ格納。_status に表示。例外は握りつぶさず記録 |
| 結果JSONが未生成 | /results で404断片を返す |
| GEMINI_API_KEY未設定でTrack A | /analyze で事前チェックし、設定方法を案内する断片を返す |
| 未登録 job_id へのアクセス | レジストリ存在チェックで404（パストラバーサル防止も兼ねる） |

---

## 8. テスト方針

| 対象 | 方法 |
|---|---|
| `routes.py` | FastAPI `TestClient` でエンドポイント単体テスト（分析関数はモック） |
| `jobs.py` | JobRegistry の状態遷移（registered→running→done/error）をテスト |
| `video_stream.py` | **Rangeヘッダ付きリクエストで 206 が返り Content-Range が正しいことをテスト**（中核要件のため必須） |
| `ids.py` | 同名ファイルの一意化、未登録ID拒否、サニタイズをテスト |
| 動画↔タイムライン同期（Alpine.js） | MVPでは自動テスト対象外（手動確認）。ロジックが薄いため |

---

## 9. スコープ

### 9.1 このフェーズに含める

- アップロード→分析→結果確認→動画照合の一連フロー
- Track A/B 切替表示
- 評価指標テーブル（正解アノテーションがある場合）
- Range対応動画ストリーム
- インメモリジョブ管理
- ライブラリのベンダリング（オンプレ対応）

### 9.2 含めない（将来フェーズ）

| 除外項目 | 理由 |
|---|---|
| ユーザー認証・ログイン | マルチユーザー化（Approach C）と同時 |
| アノテーション編集UI（結果の手修正） | 確認はできるが境界/ラベルの修正・保存は次フェーズ |
| 手順書ドラフトのインライン編集 | 次フェーズ |
| リアルタイム解析・WebSocket進捗 | 次フェーズ（現状はポーリングで十分） |
| 永続ジョブストア・ジョブ履歴一覧 | マルチユーザー化と同時 |
| **アップロード動画・結果の保持/クリーンアップ** | **既知の制約**。今フェーズは手動削除前提。溜まり続けるとディスクを圧迫するため、将来フェーズで保持期間ポリシーを導入 |

---

## 10. 依存関係の追加

既存 `pyproject.toml` に追加：

```
fastapi>=0.110
uvicorn[standard]>=0.29
jinja2>=3.1
python-multipart>=0.0.9   # multipart アップロード用
```

htmx・Alpine.js は npm 不要（`static/vendor/` に同梱）。
