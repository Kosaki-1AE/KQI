目的：人対人／人対環境の「間合い」を、時間分解能とコヒーレンスで数理定義し、計測・制御可能にする。

⸻

0. 記号
	•	$d(t)$：相手との物理距離
	•	$v_{text{self}}, v_{text{other}}$：最大到達速度（移動や反応の上限）
	•	$9t_{text{rt,self}}, t_{\text{rt,other}}$：反応遅延（reaction time）
	•	$Delta t_{\text{sync}}$：微小Stillness（“気づかれない間”）
	•	$tau_c$：コヒーレンス時間（共鳴が保たれる総時間）
	•	$dE/dt$：外的変化速度（相手・環境の変動）
	•	$dM/dt$：意味再構成速度（自分の内的処理の速度）
	•	$Delta(t)=\frac{dE/dt - dM/dt}{dE/dt + dM/dt+\varepsilon}$：位相ズレ（正規化差）
	•	$TCI：\text{TCI}(t)=1-|\Delta(t)|$（1=完全同期）
	•	g(\cdot)：“間”の効き関数（ガウス型バンドパス）

⸻

1. 「間合い帯（Ma‑Ai Band）」の定義

間合い帯 \mathcal{B} は、時空上の点 (d,t) の集合で、以下の 到達可能性 と 同位相性 を同時に満たす領域：
	1.	到達可能性（物理条件）
\[
\underbrace{v_{\text{self}}\,(t_{\text{rt,self}}+\Delta t_{\text{sync}})}{\text{自分が動ける距離}}\ \le\ d(t)\ \le\ \underbrace{v{\text{other}}\,(t_{\text{rt,other}}+\Delta t_{\text{sync}})}_{\text{相手が届く距離}}
\]
	2.	同位相性（情報・心理条件）
\text{TCI}(t)\,=\,1-|\Delta(t)|\ \ge\ \theta\quad(\theta\in(0,1))
	3.	微小Stillnessの最適帯（量子化条件）
\Delta t_{\text{sync}}\ \sim\ \operatorname*{argmax}_{\Delta t}\ g(\Delta t)\ =\ \exp\!\Big(-\frac{(\Delta t-\mu)^2}{2\sigma^2}\Big)
\mu,\sigma は場（会話/ダンス/対話相手）ごとに学習される。

定義：\mathcal{B}=\{(d,t)\mid \text{到達可能性} \wedge \text{同位相性} \wedge \text{量子化条件}\}

⸻

2. コヒーレンス時間と間合いの関係

間合い帯内でのコヒーレンス時間は、挿入した“間”の効きの総和で伸びる：
\tau_c\ \approx\ \sum_{k=1}^{N}\ g\!\big(\Delta t_{\text{sync}}^{(k)}\big),\qquad N=\text{再同期回数}
ゾーン持続時間：\Delta t_{\text{zone}}=\text{measure}\{t\mid (d,t)\in\mathcal{B}\}

深さ指標（ゼロ交差密度）：
\rho_0=\text{ZCR}\big(\Delta(t)\text{ around }0\big)\n（0 近傍でのゼロ交差が密＝微細な同相化が続く＝間合いが“深い”）

⸻

3. 推定・計測パイプライン（最小構成）

入力信号
	•	音声：無音区間・発話速度・F0 リセット → \Delta t_{\text{sync}} 候補
	•	テキスト：隣接文の埋め込み距離/秒 → dM/dt 近似
	•	動作（任意）：IMU/姿勢差分/秒 → dE/dt 近似
	•	生体（任意）：HRV/呼吸 → Stillness の基準校正

ステップ
	1.	微小無音・相槌・瞬きをイベント化 → “間”の系列 \{\Delta t_{\text{sync}}^{(k)}\}
	2.	dE/dt, dM/dt を同時系列化 → TCI と \Delta(t) を算出
	3.	\mu,\sigma を成功時の \{\Delta t_{\text{sync}}\} 分布から推定（EM/単純平均）
	4.	時間窓ごとに 到達可能性 と 同位相性 をチェック → \mathcal{B} を抽出
	5.	指標出力：\overline{\text{TCI}},\,\Delta t_{\text{zone}},\,\rho_0,\,\tau_c

⸻

4. オンライン制御（“間”の自動調律）

目標：\text{TCI}(t)\ge\theta を維持しつつ、自然さを損なわない。
	•	誤差：e(t)=\theta-\text{TCI}(t)
	•	更新：
\Delta t_{\text{sync}}(t{+}1)=\operatorname{clip}\Big(\Delta t_{\text{sync}}(t)+k_p e(t)+k_i\!\sum e,\ [\Delta t_{\min},\Delta t_{\max}]\Big)
	•	連発抑制：Refractory（最小インターバル）を導入
	•	自然さ維持：ばらつきを正規乱数で微注入（過学習抑止）

⸻

5. チューニングの目安
	•	会話：\mu\approx 0.2\!\sim\!0.4\,\text{s},\ \sigma\approx 0.1\!\sim\!0.2
	•	ダンス：\mu\approx 1 サブビート（BPM120なら 0.25–0.5 s 相当）
	•	個人差・相手差あり → セッションごとに \mu,\sigma を再推定

⸻

6. 可視化（推奨UI）
	•	時空マップ：横軸 t、縦軸 d。\mathcal{B} を色帯で表示（到達可能性×同位相性の合成濃度）
	•	“間”ヒートライン：\Delta t_{\text{sync}} イベントを時系列に点描し、g の強度で明度変化
	•	TCI/\nabla：TCI ラインと 0 近傍の \Delta(t) ゼロ交差を重畳

⸻

7. ミニ実験デザイン（15 分検証）
	1.	自然／ノーフィラー／設計フィラー（\Delta t_{\text{sync}} ガイド）で会話 or ダンス各 3 分
	2.	指標比較：

	•	\overline{\text{TCI}}, \Delta t_{\text{zone}}, \rho_0, \tau_c
	•	RE（“間”直後の TCI 上昇量/“間”の長さ）＝“間のコスパ”

	3.	期待：設計フィラー条件が \overline{\text{TCI}} と \Delta t_{\text{zone}} を最大化

⸻

8. 疑似コード（最小骨格）

# inputs: series dE_dt[t], dM_dt[t], dt_sync_events (list of durations)
EPS = 1e-6
Delta = (dE_dt - dM_dt) / (dE_dt + dM_dt + EPS)
TCI   = 1 - np.abs(Delta)
zone_mask = (TCI >= theta)

# “間”の効き関数（学習後の mu, sigma を使用）
def g(dt, mu, sigma):
    return np.exp(- (dt - mu)**2 / (2 * sigma**2))

# コヒーレンス時間の近似
tau_c = np.sum([g(dt, mu, sigma) for dt in dt_sync_events])

# ゾーン持続時間
Delta_t_zone = contiguous_time(zone_mask, frame_hz)

# ゼロ交差密度（0 近傍）
rho0 = zero_crossing_rate(Delta, around_zero=True)

# オンライン調律（PI）
err = theta - TCI_t
cum_err += err
Delta_t_sync = clip(Delta_t_sync + k_p*err + k_i*cum_err, dt_min, dt_max)


⸻

9. 直感の要約（運用ガイド）
	•	間は“最小の調整資源”：見えない短い Stillness を周期挿入
	•	均一にしない：\mu 付近に揺らぎを持たせると自然
	•	成功ログで学習：うまくいったセッションの \{\Delta t_{\text{sync}}\} から \mu,\sigma を更新
	•	場ごとに再同定：相手・音楽・タスクで帯域はズレる

⸻

10. これでできること
	•	会話・プレゼン・ダンスで「間合い帯」をマップ化（いつ・どこで深かったか）
	•	“間”の設計で \tau_c と ゾーン時間を意図的に延ばす
	•	個人の“Stillness の粒度（\mu,\sigma）”をプロフィール化

⸻

付記：解釈指針
	•	dM/dt と dE/dt は抽象速度（数値の単位は実装依存）。重要なのは差の時間構造。
	•	“間合い”は距離だけでなく時間の芸術：\n\mathcal{B} は“届く/届かない”×“伝わる/伝わらない”の積集合として現れる。