# src/models/cnn_bilstm.py
import torch
import torch.nn as nn
from typing import Dict
from.base_model import BaseModel

class CNNBiLSTM(BaseModel):
    def __init__(self, input_dim: int, config: Dict):
        super(CNNBiLSTM, self).__init__(input_dim, config)
        
        model_config = config.get('cnn_bilstm', {})
        self.hidden_dim = int(model_config.get('hidden_dim', 64))
        num_layers = int(model_config.get('num_layers', 2))
        kernel_size = int(model_config.get('kernel_size', 3))
        
        # SỬA ĐỔI: Lấy dropout 0.25 từ config mới
        dropout = float(model_config.get('dropout_rate', 0.25)) 

        # 1. CNN Block
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=self.hidden_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # SỬA ĐỔI: Dropout nhẹ sau CNN
        self.dropout_cnn = nn.Dropout(0.1) 

        # 2. BiLSTM Block
        self.bilstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 3. Output Block
        self.dropout_fc = nn.Dropout(dropout)
        self.output_head = nn.Linear(self.hidden_dim * 2, 2) # 2 classes: Up/Down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, timesteps, features) -> Permute cho CNN: (batch, features, timesteps)
        x = x.permute(0, 2, 1) 
        
        x = self.conv1(x)
        x = self.relu(x)
        
        # Chỉ pool nếu chuỗi đủ dài
        if x.shape[2] > 2:
            x = self.pool(x)
        
        x = self.dropout_cnn(x)

        # Permute lại cho LSTM: (batch, new_timesteps, hidden_dim)
        x = x.permute(0, 2, 1) 
        
        lstm_out, _ = self.bilstm(x)
        
        # Lấy hidden state của bước thời gian cuối cùng
        last_output = lstm_out[:, -1, :]
        
        last_output = self.dropout_fc(last_output)
        logits = self.output_head(last_output)
        
        return logits