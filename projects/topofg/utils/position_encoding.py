import math
import torch
import torch.nn as nn

class SeqPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 11):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # [11, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)          # [1, 11, 256]
        pe[0, :, 0::2] = torch.sin(position * div_term)  
        pe[0, :, 1::2] = torch.cos(position * div_term)  
        self.register_buffer('pe', pe)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        x = x.unsqueeze(0).unsqueeze(-1)  # x 的形状将变为 [1, 11, 1]
        seq_len = x.size(1)  # 获取输入张量的序列长度
        # 确保切片的长度不超过位置编码的最大长度
        seq_pos = self.pe[:, :min(seq_len, self.pe.size(1))]  # 修正切片长度

        return seq_pos
