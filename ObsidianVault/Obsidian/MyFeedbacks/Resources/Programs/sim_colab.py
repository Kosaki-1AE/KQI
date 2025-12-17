import math
from dataclasses import dataclass
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F


class Action(Enum):
    STAY_STILLNESS = auto()   # その層に留まる（保留）
    GO_NEXT = auto()          # 次の層へ進む
    GO_BACK = auto()          # 1つ戻る（再初期化の一部）
    ASK_CLARIFY = auto()      # 聞き直し（外部に質問）
    DECODE = auto()           # 出力してよい（生成へ）

@dataclass
class ControllerConfig:
    # 「迷い」指標（logits entropy）
    ent_high: float = 6.0     # これ以上なら迷いすぎ →止まる/聞き直し
    ent_low: float = 3.0      # これ以下なら十分自信 → decode許可候補

    # 「落ち着き」指標（Δh：層間の変化量）
    dh_small: float = 0.20    # これ以下なら落ち着いた（Stillness成立）
    dh_large: float = 0.80    # これ以上なら暴れてる（戻る/止まる）

    # 行動の優先度・安全装置
    max_stay_steps: int = 3   # 止まり続けたら聞き直しへ
    allow_decode: bool = True

class GSMCController:
    """
    metrics を見て action を返すコントローラ。
    - entropy（必須級）：迷い度
    - delta_h（任意）：落ち着き度（層変化）
    - step_in_layer（任意）：同じ層に留まった回数
    """
    def __init__(self, cfg: ControllerConfig | None = None):
        self.cfg = cfg or ControllerConfig()

    def decide(self, metrics: dict) -> Action:
        """
        metrics例:
          {
            "entropy": float or torch.Tensor scalar,
            "delta_h": float or torch.Tensor scalar (optional),
            "step_in_layer": int (optional),
            "phase": "stillness" | "coherence" | "motion" | ... (optional)
          }
        """
        ent = metrics.get("entropy", None)
        dh  = metrics.get("delta_h", None)
        stay_steps = int(metrics.get("step_in_layer", 0))
        phase = metrics.get("phase", None)

        # tensor -> float
        if isinstance(ent, torch.Tensor):
            ent = float(ent.detach().cpu().item())
        if isinstance(dh, torch.Tensor):
            dh = float(dh.detach().cpu().item())

        # 0) entropy無いと判断できないので保留
        if ent is None:
            return Action.STAY_STILLNESS

        # 1) 迷いが高すぎる：止まる or 聞き直し
        if ent >= self.cfg.ent_high:
            if stay_steps >= self.cfg.max_stay_steps:
                return Action.ASK_CLARIFY
            return Action.STAY_STILLNESS

        # 2) Δhが取れている場合の安全装置
        if dh is not None:
            # 暴れてる：戻る or 止める
            if dh >= self.cfg.dh_large:
                return Action.GO_BACK
            # 落ち着いた：次へ進む（Stillness成立）
            if dh <= self.cfg.dh_small:
                # Coherenceフェーズなら decode 判定へ寄せる
                if phase == "coherence" and self.cfg.allow_decode and ent <= self.cfg.ent_low:
                    return Action.DECODE
                return Action.GO_NEXT

        # 3) entropyが十分低い（自信がある）なら decode（ただし許可制）
        if self.cfg.allow_decode and ent <= self.cfg.ent_low:
            # phaseが指定されてるなら coherence でのみ decode するのが安全
            if phase is None or phase == "coherence":
                return Action.DECODE
            # coherence以外なら一旦次へ（出力は保留）
            return Action.GO_NEXT

        # 4) 中間：とりあえず次へ
        return Action.GO_NEXT

def last_token_entropy(logits: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    # logits: (B, T, V)
    p = torch.softmax(logits[:, -1, :], dim=-1)          # (B, V)
    ent = -(p * (p + eps).log()).sum(dim=-1)             # (B,)
    return ent.mean()  # とりまバッチ平均（scalar）

def token_entropy_from_logits(logits: torch.Tensor, eps: float = 1e-9):
    p = torch.softmax(logits, dim=-1)
    ent = -(p * (p + eps).log()).sum(dim=-1)   # (B, T)
    return ent

# ----------------------------
# Positional Encoding　経験則をまんま持ってきたものがこれ。ようやくできるようになったわ。
# ----------------------------
class PositionalEncoding(nn.Module): #意味のある位置情報を埋め込みに追加する
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ----------------------------
# Masks
# ----------------------------
def generate_square_subsequent_mask(sz: int, device=None) -> torch.Tensor: #注目すべき未来の情報を隠すためのマスク
    """
    Causal mask for decoder self-attn.
    Returns (sz, sz) where upper triangle is -inf.
    """
    device = device or torch.device("cpu")
    mask = torch.full((sz, sz), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask

def make_padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    tokens: (batch, seq_len)
    returns: (batch, seq_len) True where PAD
    """
    return tokens.eq(pad_id)

def layer_delta_h(hiddens: list[torch.Tensor]) -> torch.Tensor:
    """
    hiddens: list of (B, L, D)
    returns: scalar tensor（平均Δh）
    """
    if len(hiddens) < 2:
        return torch.tensor(0.0, device=hiddens[0].device)
    dh = []
    for i in range(1, len(hiddens)):
        # (B,L,D) -> まずDのノルム → (B,L) → 平均
        d = (hiddens[i] - hiddens[i-1]).norm(p=2, dim=-1).mean()
        dh.append(d)
    return torch.stack(dh).mean()

def train_step(model, optimizer, src, tgt, pad_id=0):
    """
    src: (B, S)
    tgt: (B, T)  ※BOS込みの想定でも、無い想定でもOK（ここではシンプルにランダム）
    """
    model.train()
    optimizer.zero_grad()

    # teacher forcing 用に 1個ずらす
    tgt_in  = tgt[:, :-1]   # (B, T-1)
    tgt_out = tgt[:, 1:]    # (B, T-1)

    logits = model(src, tgt_in)  # (B, T-1, V)

    # PADはlossから除外
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt_out.reshape(-1),
        ignore_index=pad_id
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 任意だけど安定する
    optimizer.step()
    return loss.item(), logits

@torch.no_grad()
def measure_metrics(model, src, tgt_in, pad_id=0):
    model.eval()

    device = src.device
    src_pad = make_padding_mask(src, pad_id)
    tgt_pad = make_padding_mask(tgt_in, pad_id)
    tgt_mask = generate_square_subsequent_mask(tgt_in.size(1), device=device)

    memory, _ = model.encode_with_layers(src, src_key_padding_mask=src_pad)
    dec_out, dec_h = model.decode_with_layers(
        tgt_in, memory,
        tgt_mask=tgt_mask,
        tgt_key_padding_mask=tgt_pad,
        memory_key_padding_mask=src_pad,
    )

    logits = model.generator(dec_out)
    ent_last_mean = last_token_entropy_ratio(logits)      # scalar tensor
    dh_dec = layer_delta_h(dec_h)                   # scalar tensor
    return float(ent_last_mean), float(dh_dec), logits.shape

def last_token_entropy_ratio(logits, eps=1e-9):
    V = logits.size(-1)
    p = torch.softmax(logits[:, -1, :], dim=-1)
    ent = -(p * (p + eps).log()).sum(dim=-1).mean()
    return float(ent) / math.log(V)

def dh_ratio(dh: float, d_model: int) -> float:
    return float(dh) / math.sqrt(d_model)   # 雑にスケール合わせ

def make_batch_random(B, src_len, tgt_len, src_vocab, tgt_vocab, pad_id=0, device="cpu"):
    src = torch.randint(1, src_vocab, (B, src_len), device=device)
    tgt = torch.randint(1, tgt_vocab, (B, tgt_len), device=device)
    src[:, -1] = pad_id
    tgt[:, -1] = pad_id
    return src, tgt

def make_batch_copy(B, src_len, tgt_len, vocab, pad_id=0, device="cpu"):
    # srcを作って、tgtは「srcのコピー」（最小の“信じていい世界”）
    src = torch.randint(1, vocab, (B, src_len), device=device)
    src[:, -1] = pad_id

    tgt = torch.full((B, tgt_len), pad_id, device=device, dtype=torch.long)
    L = min(src_len, tgt_len)
    tgt[:, :L] = src[:, :L]
    return src, tgt

# ----------------------------
# Full Transformer (Embedding -> Encoder -> Decoder -> Linear)
# ----------------------------
class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_id: int = 0,
        share_embeddings: bool = False,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        # Embeddings
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        if share_embeddings and (src_vocab_size == tgt_vocab_size):
            self.tgt_tok_emb = self.src_tok_emb
        else:
            self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # ---------
        # Encoder/Decoder layers (batch_first=True)
        # ---------
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=False,
            )
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=False,
            )
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # Output head
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Optional: tie output with target embedding
        self.tie_output = False
        if share_embeddings and (src_vocab_size == tgt_vocab_size):
            self.tie_output = True
            self.generator.weight = self.tgt_tok_emb.weight

    # ---- helpers ----
    def _embed_src(self, src_tokens: torch.Tensor) -> torch.Tensor:
        src = self.src_tok_emb(src_tokens) * math.sqrt(self.d_model)
        return self.pos_enc(src)

    def _embed_tgt(self, tgt_tokens: torch.Tensor) -> torch.Tensor:
        tgt = self.tgt_tok_emb(tgt_tokens) * math.sqrt(self.d_model)
        return self.pos_enc(tgt)

    # ---- with layer traces ----
    def encode_with_layers(self, src_tokens: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None):
        """
        returns:
          memory: (B, S, D)
          enc_hiddens: list[(B,S,D)]  各層の出力
        """
        x = self._embed_src(src_tokens)
        enc_hiddens = []
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
            enc_hiddens.append(x)
        x = self.encoder_norm(x)
        return x, enc_hiddens

    def decode_with_layers(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ):
        """
        returns:
          out: (B, T, D)
          dec_hiddens: list[(B,T,D)] 各層の出力
        """
        y = self._embed_tgt(tgt_tokens)
        dec_hiddens = []
        for layer in self.decoder_layers:
            y = layer(
                y,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            dec_hiddens.append(y)
        y = self.decoder_norm(y)
        return y, dec_hiddens

    # ---- original APIs (keep) ----
    def encode(self, src_tokens: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None):
        memory, _ = self.encode_with_layers(src_tokens, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ):
        out, _ = self.decode_with_layers(
            tgt_tokens, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return out

    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor):
        device = src_tokens.device

        src_pad = make_padding_mask(src_tokens, self.pad_id)  # (B,S)
        tgt_pad = make_padding_mask(tgt_tokens, self.pad_id)  # (B,T)

        tgt_len = tgt_tokens.size(1)
        tgt_mask = generate_square_subsequent_mask(tgt_len, device=device)  # (T,T)

        memory = self.encode(src_tokens, src_key_padding_mask=src_pad)
        dec_out = self.decode(
            tgt_tokens,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        logits = self.generator(dec_out)
        return logits

# ----------------------------
# Quick sanity check
# ----------------------------
if __name__ == "__main__":
    B, src_len, tgt_len = 2, 7, 6
    src_vocab, tgt_vocab = 1000, 1200
    pad_id = 0

    model = TransformerSeq2Seq(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        pad_id=pad_id,
        share_embeddings=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    controller = GSMCController()

    src = torch.randint(1, src_vocab, (B, src_len))
    tgt_in = torch.randint(1, tgt_vocab, (B, tgt_len))
    src[:, -1] = pad_id
    tgt_in[:, -1] = pad_id

    # ===== 固定の観測用データ（probe）を作る =====
    probe_src = torch.randint(1, src_vocab, (B, src_len))
    probe_tgt = torch.randint(1, tgt_vocab, (B, tgt_len))
    probe_src[:, -1] = pad_id
    probe_tgt[:, -1] = pad_id
    probe_tgt_in = probe_tgt[:, :-1]  # decoder入力用

    # ===== 学習前の基準値 =====
    base_ent, base_dh, shape0 = measure_metrics(model, probe_src, probe_tgt_in, pad_id)
    base_action = controller.decide({"entropy": base_ent, "delta_h": base_dh, "phase": "coherence", "step_in_layer": 0})
    print(f"[BASE] logits={shape0} ent={base_ent:.3f} dh={base_dh:.3f} action={base_action}")

    # ===== 学習ループ =====
    for step in range(1, 201):
        # 学習用のランダムデータ（ここは後で本物データに置き換えればOK）
        src = torch.randint(1, src_vocab, (B, src_len))
        tgt = torch.randint(1, tgt_vocab, (B, tgt_len))
        src[:, -1] = pad_id
        tgt[:, -1] = pad_id

        loss, _ = train_step(model, optimizer, src, tgt, pad_id=pad_id)

        # ===== 10ステップごとに “同じprobe” で現在値を観測 =====
        if step % 10 == 0:
            cur_ent, cur_dh, shape = measure_metrics(model, probe_src, probe_tgt_in, pad_id)
            cur_action = controller.decide({"entropy": cur_ent, "delta_h": cur_dh, "phase": "coherence", "step_in_layer": 0})

            print(
                f"[STEP {step:03d}] loss={loss:.3f} "
                f"ent={cur_ent:.3f} (Δ{cur_ent-base_ent:+.3f}) "
                f"dh={cur_dh:.3f} (Δ{cur_dh-base_dh:+.3f}) "
                f"action={cur_action}"
            )
