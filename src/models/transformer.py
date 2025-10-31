# src/models/transformer.py
import torch
import torch.nn as nn
import math
from typing import Dict

# --- THAY ĐỔI: Import BaseModel ---
from .base_model import BaseModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# --- THAY ĐỔI: Kế thừa từ BaseModel ---
class TransformerModel(BaseModel):
    def __init__(self, input_dim: int, config: Dict):
        # --- THAY ĐỔI: Gọi super().__init__ ---
        super(TransformerModel, self).__init__(input_dim, config)
        
        model_config = config.get('transformer', {})
        d_model = model_config.get('d_model', 128)
        nhead = model_config.get('nhead', 4)
        num_layers = model_config.get('num_layers', 2)
        dim_feedforward = model_config.get('dim_feedforward', 256)
        dropout = model_config.get('dropout_rate', 0.2)

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.d_model = d_model
        
        # --- THAY ĐỔI: Tạo đầu ra động ---
        self.output_heads = nn.ModuleDict()
        
        # self.horizons đến từ BaseModel, có thể là ['1','3','5'] hoặc ['1Q']
        for h in self.horizons:
            # Đầu ra Hồi quy
            self.output_heads[f'Target_Close_t+{h}'] = nn.Linear(d_model, 1)
            # Đầu ra Phân loại
            self.output_heads[f'Target_Direction_t+{h}'] = nn.Linear(d_model, 2)
        # --- KẾT THÚC THAY ĐỔI ---

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x shape: (batch_size, timesteps, features)
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = x.permute(1, 0, 2) # (timesteps, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2) # (batch_size, timesteps, d_model)
        
        x = self.transformer_encoder(x)
        
        # Lấy output của bước thời gian cuối cùng
        last_output = x[:, -1, :] 
        
        # --- THAY ĐỔI: Đưa qua tất cả các đầu ra ---
        outputs = {}
        for h in self.horizons:
            outputs[f'Target_Close_t+{h}'] = self.output_heads[f'Target_Close_t+{h}'](last_output)
            outputs[f'Target_Direction_t+{h}'] = self.output_heads[f'Target_Direction_t+{h}'](last_output)
            
        return outputs
        # --- KẾT THÚC THAY ĐỔI ---