# src/models/transformer.py
import torch
import torch.nn as nn
import math
from typing import Dict
from.base_model import BaseModel, PositionalEncoding

class TransformerModel(BaseModel):
    def __init__(self, input_dim: int, config: Dict):
        super(TransformerModel, self).__init__(input_dim, config)
        
        model_config = config.get('transformer', {})
        d_model = int(model_config.get('d_model', 64))
        nhead = int(model_config.get('nhead', 4))
        num_layers = int(model_config.get('num_layers', 2))
        dim_feedforward = int(model_config.get('dim_feedforward', 128))
        # SỬA ĐỔI: Nhận giá trị 0.25 từ config mới
        dropout = float(model_config.get('dropout_rate', 0.25))

        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True 
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.final_dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        
        # 1. Project & Scale
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # 2. Positional Encoding
        # PE expect (Seq, Batch, Feature) -> Permute
        x = x.permute(1, 0, 2) 
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2) # Trả lại (Batch, Seq, Feat)
        
        # 3. Transformer Encoder
        x = self.transformer_encoder(x)
        
        # 4. Lấy phần tử cuối cùng
        last_output = x[:, -1, :] 
        
        # 5. Output
        last_output = self.final_dropout(last_output)
        logits = self.output_head(last_output)
        
        return logits