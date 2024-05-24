import torch
import data_prep.Yujie_Guo.config_img as config

import torchvision.models as models
import torch.nn as nn
import math
from torch import Tensor
from typing import Dict
import numpy as np
import timm
import copy
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class Resnet_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self._init_resnet_model()

    def forward(self, images):
        images = images.reshape(-1, 3, config.image_height, config.image_width)
        return self.resnet_model(images)

    def resnet_model(self, x):
        x = self.resnet50_model(x)
        x = self.resnet50_model_final_fc(x)
        batch_size = x.size(0) // config.n_input_image_frames
        return x.view(batch_size, -1, config.embedding_dim)

    def _init_resnet_model(self):
        self.resnet50_model = models.resnet50(pretrained=True)
        # 修改第一个卷积层以适应输入图像大小（如果需要的话）
        self.resnet50_model.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)

        # 冻结 ResNet50 的预训练层
        for param in self.resnet50_model.parameters():
            param.requires_grad = False

        # 添加并训练新的全连接层
        self.resnet50_model_final_fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, config.embedding_dim)
        )

        # 确保新添加的层是可训练的
        for param in self.resnet50_model_final_fc.parameters():
            param.requires_grad = True

class Ego_state_embedding(nn.Module):
    """
    ego state embedding
    """
    def __init__(self, input_size :int = config.trajectory_xy_dim , emb_size :int=config.embedding_dim):
        super(Ego_state_embedding, self).__init__()
        self.embedding = nn.Linear(input_size, emb_size)    
    def forward(self, tgt: Tensor):
        return self.embedding(tgt)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = config.embedding_dim, dropout: float = 0.1, max_length: int = 5000):
        super().__init__()
        """
        Args:
          d_model:      dimension of embeddings
          dropout:      randomly zeroes-out some of the input
          max_length:   max sequence length
        """
        # initialize dropout
        self.dropout = nn.Dropout(p=dropout)
        # create tensor of 0s
        pe = torch.zeros(max_length, d_model)
        # create position column
        k = torch.arange(0, max_length).unsqueeze(1)
        # calc divisor for positional encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # calc sine on even indices
        pe[:, 0::2] = torch.sin(k * div_term)
        # calc cosine on odd indicesal / len(val_dataloader):.3f}')
        pe = pe.unsqueeze(0)
        # buffers are saved in state_dict but not trained by the optimizer
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        """
        Args:
          x:        embeddings (batch_size, seq_length, d_model)

        Returns:
                    embeddings + positional encodings (batch_size, seq_length, d_model)
        """
        # add positional encoding to the embeddings
        x = x + self.pe[:, : x.size(1)].requires_grad_(False) # type: ignore
        # perform dropout
        return self.dropout(x)

class Target_embedding(nn.Module):
    """
    target embedding
    """
    def __init__(self, input_size :int = config.trajectory_xy_dim , emb_size :int=config.embedding_dim):
        super(Target_embedding, self).__init__()
        self.embedding = nn.Linear(input_size, emb_size)    
    def forward(self, tgt: Tensor):
        return self.embedding(tgt)
    
class TransformerModel(nn.Module):
    def __init__(self,
                 d_model: int = config.embedding_dim,
                 nhead: int = 8,
                 nlayers: int = 1,
                 dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout, 5000)
        self.ego_state_embedding = Ego_state_embedding(config.trajectory_xy_dim, config.embedding_dim)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layers, nlayers)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.decoder = TransformerDecoder(decoder_layers, nlayers)
        self.backbone = Resnet_backbone()
        self.tgt_embedding = Target_embedding(config.trajectory_xy_dim, config.embedding_dim)
        self.fc1 = nn.Linear(config.embedding_dim, config.trajectory_xy_dim)
        self.lstm_decoder = nn.LSTM(input_size=config.embedding_dim, hidden_size=config.embedding_dim, num_layers=5, batch_first=True)
        self.fc2 = nn.Linear(config.n_input_frames*config.embedding_dim, config.n_output_frames*config.trajectory_xy_dim)
        self.gru_decoder = nn.GRU(input_size=config.embedding_dim, hidden_size=config.embedding_dim)

    def forward(self, images, ego_state, tgt):
        images = self.backbone(images)
        images = self.pos_encoder(images)
        coors = self.ego_state_embedding(ego_state)
        coors = self.pos_encoder(coors)
        # element-wise multiplication
        src = torch.mul(images, coors)
        tgt_len = tgt.size()[1]
        
        tgt_embedded = self.tgt_embedding(tgt)
        tgt = self.pos_encoder(tgt_embedded)
        
        device = tgt.device
        tgt_mask = torch.triu(torch.full((tgt_len, tgt_len), float('-inf'), dtype=torch.get_default_dtype(), device=device), diagonal=1)
        
        memory = self.encoder(src)
        output = self.decoder(tgt, memory, tgt_mask)
        
        output = self.fc1(output)
        return output

    def lstm_decode(self, images, ego_state):
        images = self.backbone(images)
        images = self.pos_encoder(images)
        coors = self.ego_state_embedding(ego_state)
        coors = self.pos_encoder(coors)
        src = torch.mul(images, coors)
        memory = self.encoder(src)
        output, (h_0, c_0) = self.lstm_decoder(memory)
        bs = output.size()[0]
        output = output.view(bs, -1)
        output = self.fc2(output)
        output = output.view(bs, -1, config.trajectory_xy_dim)
        return output

    def gru_decode(self, images, ego_state):
        # 提取特征并应用位置编码
        images = self.backbone(images)
        images = self.pos_encoder(images)
        coors = self.ego_state_embedding(ego_state)
        coors = self.pos_encoder(coors)

        # 元素级乘法合并特征
        src = torch.mul(images, coors)

        # 使用编码器处理合并后的特征
        memory = self.encoder(src)

        # GRU 解码过程
        # 由于 GRU 不需要单独的细胞状态，只需要一个初始隐藏状态即可，这里我们让 PyTorch 自动初始化这个状态
        output, h_n = self.gru_decoder(memory)

        # 由于可能有多层，取最后一层的隐藏状态作为输出特征
        # 将 GRU 的输出通过一个线性层以匹配最终输出的预期维度
        bs = output.size()[0]
        output = output.view(bs, -1)
        output = self.fc2(output)
        output = output.view(bs, -1, config.trajectory_xy_dim)
        return output















