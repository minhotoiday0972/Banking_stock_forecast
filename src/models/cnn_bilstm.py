# src/models/cnn_bilstm.py
import torch
import torch.nn as nn
from typing import Dict

# --- THAY ĐỔI: Import BaseModel ---
from .base_model import BaseModel

# --- THAY ĐỔI: Kế thừa từ BaseModel ---
class CNNBiLSTM(BaseModel):
    def __init__(self, input_dim: int, config: Dict):
        # --- THAY ĐỔI: Gọi super().__init__ ---
        super(CNNBiLSTM, self).__init__(input_dim, config)
        
        model_config = config.get('cnn_bilstm', {})
        self.hidden_dim = model_config.get('hidden_dim', 64)
        num_layers = model_config.get('num_layers', 2)
        kernel_size = model_config.get('kernel_size', 3)
        dropout = model_config.get('dropout_rate', 0.5)

        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=self.hidden_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.bilstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # --- THAY ĐỔI: Tạo đầu ra động ---
        self.output_heads = nn.ModuleDict()
        
        # self.horizons đến từ BaseModel, có thể là ['1','3','5'] hoặc ['1Q']
        for h in self.horizons:
            # Đầu ra Hồi quy (sẽ bị bỏ qua nếu loss_weight=0)
            self.output_heads[f'Target_Close_t+{h}'] = nn.Linear(self.hidden_dim * 2, 1)
            # Đầu ra Phân loại
            self.output_heads[f'Target_Direction_t+{h}'] = nn.Linear(self.hidden_dim * 2, 2)
        # --- KẾT THÚC THAY ĐỔI ---

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x shape: (batch_size, timesteps, features)
        x = x.permute(0, 2, 1) # (batch_size, features, timesteps)
        
        x = self.conv1(x)
        x = self.relu(x)
        # Bỏ qua MaxPool nếu chuỗi quá ngắn
        if x.shape[2] > 2:
             x = self.pool(x)
        
        x = x.permute(0, 2, 1) # (batch_size, timesteps_pooled, hidden_dim)
        
        lstm_out, (h_n, c_n) = self.bilstm(x)
        
        # Lấy output cuối cùng
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # --- THAY ĐỔI: Đưa qua tất cả các đầu ra ---
        outputs = {}
        for h in self.horizons:
            outputs[f'Target_Close_t+{h}'] = self.output_heads[f'Target_Close_t+{h}'](last_output)
            outputs[f'Target_Direction_t+{h}'] = self.output_heads[f'Target_Direction_t+{h}'](last_output)
        
        return outputs
        # --- KẾT THÚC THAY ĐỔI ---