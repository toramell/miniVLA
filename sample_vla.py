import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer, CLIPVisionModel

# ======================================
# 1.Vision Encoder
# ======================================

class CLIPVisionEncoder(nn.Module):
    def __init__(self,model_name="openai/clip-vit-base-patch32", trainable=True):
        super().__init__()
        self.clip_vision = CLIPVisionModel.from_pretrained(model_name)
        self.embed_dim =  self.clip_vision.config.hidden_size
        self.image_size = self.clip_vision.config.image_size
    
        if not trainable:
            for param in self.clip_vision.parameters():
                param.requires_grad = False
        
        self.resize = nn.Upsample(
            size = (self.image_size,self.image_size),
            mode = "bilinear",
            align_corners=False
        )

        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1))
    
    def forward(self, images):
        """画像を埋め込みベクトルに変換
        image: (B, 3, H, W)
        return: (B, N, D)  N はパッチ数, D は埋め込み次元
        """
        # images: (B, 3, H, W) を CLIP ViT 入力サイズに変換
        x = self.resize(images)

        x = (x - self.mean) / self.std

        outputs = self.clip_vision(pixel_values=x)
        # last_hidden_state: (B, 1 + N_patches, D_v)
        # 先頭の CLS トークンは除いてパッチトークンのみ使う
        vision_tokens = outputs.last_hidden_state[:,1:,:]

        return vision_tokens

# ======================================
# 2. Vision → LLM Projector
# ======================================  

class MultiLayerQFormer(nn.Module):
    """
    Vision tokens + learnable query tokens を
    multi-layer TransformerEncoder で処理する Q-Former 風モジュール

    入力:  vision_tokens (B, N_v, D_v)
    出力:  q_out        (B, N_q, D_llm)
    """
    def __init__(
            self,
            vision_dim:int,
            llm_dim:int,
            num_queries:int=16,
            num_heads:int=8,
            num_layers:int=4,
            dropout:float=0.1
    ):
        super().__init__()

        if vision_dim != llm_dim:
            self.vision_proj = nn.Linear(vision_dim, llm_dim)
        else:
            self.vision_proj = nn.Identity()

        self.num_queries = num_queries
        self.llm_dim = llm_dim
        self.query_tokens = nn.Parameter(
            torch.randn(num_queries, llm_dim)
        )

        self.crossattention = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.cross_ln = nn.LayerNorm(llm_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = llm_dim,
            nhead = num_heads,
            dim_feedforward = llm_dim * 4,
            dropout = dropout,
            batch_first = True,
            activation = "gelu"
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, vision_tokens:torch.Tensor) -> torch.Tensor:
        """
        Vision tokens: (B ,N_v,D_v)
        return: (B , N_q, D_llm)
        """
        B, N_v, D_v = vision_tokens.shape

        # 1. vision を llm_dim に射影
        v = self.vision_proj(vision_tokens)

        # 2. query_tokens をバッチに複製
        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        
        # 3. まず cross-attn で vision から情報を引き出す
        #    Q = query_tokens, K=V=vision_tokens
        q2 , _ =self.crossattention(q, v, v) #(B ,N_q ,D_llm)
        q = self.cross_ln(q + q2)

        # 4. その後、multi-layer TransformerEncoder で query を refine
        #    Encoder は self-attn だけ（vision は使わない）
        q_out = self.encoder(q)  # (B, N_q, D_llm)

        return q_out

 
class QFormerFusion(nn.Module):
    def __init__(
            self,
            vision_dim,
            llm_dim,
            num_queries=16,
            num_heads=8
    ):
        super().__init__()

        if vision_dim != llm_dim:
            self.vision_proj = nn.Linear(vision_dim, llm_dim)
        else:
            self.vision_proj = nn.Identity()

        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(
            torch.randn(num_queries, llm_dim)
        )


        self.crossattention = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            batch_first=True
        )   

        self.ln = nn.LayerNorm(llm_dim)

    def forward(self, vision_tokens):
        B, N_v, D_v = vision_tokens.shape

        v = self.vision_proj(vision_tokens)

        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        
        out, _ = self.crossattention(q, v, v)
        out = self.ln(out)
        return out
# ======================================
# MiniVLA Model
# ======================================
# 
class MiniVLA(nn.Module):
    def __init__(
            self, 
            llm_name="gpt2",
            num_actions=3,
            clip_name="openai/clip-vit-base-patch16",
            qformer_num_queries=16, 
            qformer_num_heads=8,
            qformer_num_layers=4,
            vision_trainable=True,
            llm_trainable=True,
            use_llm=True  # ← LLMを使うかどうかのフラグを追加
        ):
        super().__init__()
        
        self.use_llm = use_llm

        # ---- 1. Tokenizer ----
        self.tokenizer = GPT2Tokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # ---- 2. Vision encoder（学習可能に変更） ----
        self.vision_encoder = CLIPVisionEncoder(
            model_name=clip_name, 
            trainable=vision_trainable
        )
        vision_dim = self.vision_encoder.embed_dim

        if self.use_llm:
            # ---- 3. LLM をロード ----
            self.llm = GPT2Model.from_pretrained(llm_name)
            self.llm.resize_token_embeddings(len(self.tokenizer))
            llm_dim = self.llm.config.hidden_size
            
            # LLMの学習設定（最後の4層のみ）
            if not llm_trainable:
                for param in self.llm.parameters():
                    param.requires_grad = False
            else:
                for i, block in enumerate(self.llm.h):
                    if i < len(self.llm.h) - 4:
                        for param in block.parameters():
                            param.requires_grad = False

            # ---- 4. Q-Former ----
            self.qformer = MultiLayerQFormer(
                vision_dim=vision_dim,
                llm_dim=llm_dim,
                num_queries=qformer_num_queries,
                num_heads=qformer_num_heads,
                num_layers=qformer_num_layers,
            )
            
            # ---- 5. Action head（LLM出力から） ----
            self.action_head = nn.Sequential(
                nn.Linear(llm_dim, llm_dim),
                nn.LayerNorm(llm_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(llm_dim, num_actions)
            )
            
            # ---- 6. BOS token ----
            self.bos = self.tokenizer.bos_token_id
            if self.bos is None:
                self.bos = self.tokenizer.eos_token_id
        else:
            # LLM無しモード（ベースラインと同じ構成）
            self.llm = None
            self.qformer = None
            self.bos = None
            
            # Vision tokens を直接使う
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.action_head = nn.Sequential(
                nn.Linear(vision_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_actions)
            )

    def forward(self, images, input_ids=None, attention_mask=None):
        B = images.size(0)
        
        # Vision → tokens
        vision_tokens = self.vision_encoder(images)
        
        if not self.use_llm:
            # LLM無しモード: ベースラインと同じ処理
            pooled = self.pool(vision_tokens.transpose(1, 2)).squeeze(-1)
            logits = self.action_head(pooled)
            return logits
        
        # LLMありモード
        device = input_ids.device if input_ids is not None else images.device
        
        # Q-Formerで圧縮
        qtoken = self.qformer(vision_tokens)
        
        if input_ids is None:
            # テキスト入力がない場合は、Vision tokensのみを使用
            # BOS + vision tokens
            bos_ids = torch.tensor([self.bos]).to(device)
            bos_emb = self.llm.get_input_embeddings()(bos_ids)
            bos_emb = bos_emb.expand(B, -1, -1)
            
            fused = torch.cat([bos_emb, qtoken], dim=1)
        else:
            # テキスト入力がある場合
            input_embeds = self.llm.get_input_embeddings()(input_ids)
            
            # BOS + vision + text
            bos_ids = torch.tensor([self.bos]).to(device)
            bos_emb = self.llm.get_input_embeddings()(bos_ids)
            bos_emb = bos_emb.expand(B, -1, -1)
            
            fused = torch.cat([bos_emb, qtoken, input_embeds], dim=1)

        # LLM forward
        outputs = self.llm(inputs_embeds=fused)
        
        # Vision tokensに対応する部分を使用（BOS + Q-Former tokens）
        # テキストの影響を受けにくくするため、vision部分の平均を取る
        vision_length = 1 + qtoken.size(1)  # BOS + Q-Former tokens
        vision_hidden = outputs.last_hidden_state[:, :vision_length, :]  # (B, vision_length, D)
        
        # 平均プーリング
        pooled_hidden = vision_hidden.mean(dim=1)  # (B, D)

        # Action head
        logits = self.action_head(pooled_hidden)
        return logits
    
    def get_trainable_params(self):
        """学習可能なパラメータ数を表示"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"学習可能パラメータ: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        return trainable, total

    
# ============================================================
# 4. テスト実行
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniVLA().to(device)

    # ダミー入力
    dummy_img = torch.randn(2, 3, 32, 32).to(device)
    dummy_text = ["pick the block", "move right"]

    tokenizer = model.tokenizer
    tok = tokenizer(dummy_text, return_tensors="pt", padding=True).to(device)

    logits = model(dummy_img, tok.input_ids)

    print("logits:", logits)
    print("action probs:", F.softmax(logits, dim=-1))