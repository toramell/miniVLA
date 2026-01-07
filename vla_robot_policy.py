import torch
import torch.nn as nn
import numpy as np
from sample_vla import MiniVLA, CLIPVisionEncoder, MultiLayerQFormer
from transformers import GPT2Model, GPT2Tokenizer

class VLARobotPolicy(nn.Module):
    """
    VLAモデルのロボット制御ポリシー
    - Vision: CLIP (学習可能)
    - Language: GPT-2 (凍結)
    - Fusion: Vision + Text Concatenation
    - Action: 
        - Position (3): Tanh (-1 to 1)
        - Rotation (6): Continuous 6D (Raw)
        - Gripper (1): Logits (Raw)
    """
    def __init__(
        self,
        pretrained_checkpoint=None,
        action_dim=7, # 後方互換性のため残すが、内部では9次元(3+6)を使用
        clip_name="openai/clip-vit-base-patch16",
        llm_name="gpt2",
        freeze_vision=False,
        freeze_llm=True,
        qformer_num_queries=16,
        hidden_dim=512
    ):
        super().__init__()
        
        # 1. Tokenizer & LLM
        self.tokenizer = GPT2Tokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.llm = GPT2Model.from_pretrained(llm_name)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        llm_dim = self.llm.config.hidden_size
        
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
                
        # 2. Vision Encoder & Q-Former
        self.vision_encoder = CLIPVisionEncoder(model_name=clip_name, trainable=not freeze_vision)
        vision_dim = self.vision_encoder.embed_dim
        
        self.qformer = MultiLayerQFormer(
            vision_dim=vision_dim,
            llm_dim=llm_dim,
            num_queries=qformer_num_queries
        )
        # Q-Formerは常に学習
        for param in self.qformer.parameters():
            param.requires_grad = True

        # 3. 融合とアクション生成のためのプロジェクション
        # 視覚(llm_dim) + 言語(llm_dim) = 2 * llm_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(llm_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 4. Action Heads (分離型)
        # アーム制御: 位置(3) + 回転6D(6) = 9次元
        self.arm_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 9) 
        )
        
        # グリッパー制御: 開閉(1)
        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1) 
        )

        self._initialize_weights()
        
        if pretrained_checkpoint:
            self.load_pretrained_weights(pretrained_checkpoint)

    def _initialize_weights(self):
        # Action Head周りの初期化
        for m in [self.fusion_proj, self.arm_head, self.gripper_head]:
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def load_pretrained_weights(self, checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)} layers.")

    def forward(self, images, instructions, attention_mask=None):
        """
        Args:
            images: (B, 3, H, W)
            instructions: (B, Seq) Tensor or List[str]
            attention_mask: (B, Seq) Tensor or None
        Returns:
            torch.Tensor: (B, 10) -> [pos(3), rot6d(6), gripper(1)]
        """
        device = images.device
        B = images.size(0)

        # --- Vision Processing ---
        vision_tokens = self.vision_encoder(images)
        q_tokens = self.qformer(vision_tokens) 
        
        # --- Text Processing ---
        # A. 学習時: すでにトークン化されたTensorが来る
        if isinstance(instructions, torch.Tensor):
            input_ids = instructions
            if attention_mask is None:
                # マスクがない場合はパディング以外を有効とする
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # B. 推論時: リストが来るのでここでトークン化
        else:
            tokens = self.tokenizer(
                instructions, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(device)
            input_ids = tokens.input_ids
            attention_mask = tokens.attention_mask

        # 安全策: バッチサイズが一致しない場合（DataParallelのバグ対策）
        if input_ids.size(0) != B:
            input_ids = input_ids[:B]
            attention_mask = attention_mask[:B]
        
        text_embeds = self.llm.get_input_embeddings()(input_ids) 
        
        # --- LLM Input Construction ---
        bos_tokens = torch.tensor([self.tokenizer.bos_token_id] * B, device=device).unsqueeze(1)
        bos_embeds = self.llm.get_input_embeddings()(bos_tokens)
        
        inputs_embeds = torch.cat([bos_embeds, q_tokens, text_embeds], dim=1)
        
        vision_mask = torch.ones((B, 1 + q_tokens.size(1)), device=device)
        full_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # --- LLM Forward ---
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=full_mask)
        last_hidden = outputs.last_hidden_state
        
        # --- Feature Pooling ---
        vision_start = 1
        vision_end = 1 + q_tokens.size(1)
        pooled_vision = last_hidden[:, vision_start:vision_end].mean(dim=1)
        
        text_hidden = last_hidden[:, vision_end:]
        mask_expanded = attention_mask.unsqueeze(-1)
        sum_text = (text_hidden * mask_expanded).sum(dim=1)
        count_text = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled_text = sum_text / count_text
        
        # Concatenation (Vision + Text)
        fused_features = torch.cat([pooled_vision, pooled_text], dim=-1)
        
        features = self.fusion_proj(fused_features)
        
        # --- Action Prediction ---
        arm_out = self.arm_head(features) # (B, 9)
        
        # 位置(0-2)は -1~1 に制限したいので Tanh
        pos = torch.tanh(arm_out[:, :3])
        
        # 回転(3-8)は 6D表現なので Tanh しない（生のまま）
        rot6d = arm_out[:, 3:9]
        
        # アーム結合
        arm_action = torch.cat([pos, rot6d], dim=-1)
        
        # グリッパー (Raw Logits)
        gripper_logits = self.gripper_head(features)
        
        # 最終出力: (B, 10)
        return torch.cat([arm_action, gripper_logits], dim=-1)
    
    def get_trainable_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {trainable:,}")
        return trainable