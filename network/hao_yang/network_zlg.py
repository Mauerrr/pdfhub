import torch
from torch import nn
from torch.nn import TransformerEncoder as Encoder
from torch.nn import TransformerDecoder as Decoder
from torch.nn import TransformerEncoderLayer as EncoderLayer
from torch.nn import TransformerDecoderLayer as DecoderLayer

import math
import numpy as np
import train.hao_yang.train_config as config

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# 位置编码
def position_encode(emb_size, seq_len):
    den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
    pos = torch.arange(0, seq_len).reshape(seq_len, 1)
    pos_embedding = torch.zeros((seq_len, emb_size))
    pos_embedding[:, 0::2] = torch.sin(pos * den)
    pos_embedding[:, 1::2] = torch.cos(pos * den)
    return pos_embedding

# Transformer网络
class TFMModel(nn.Module):
    def __init__(self, seq_len_in=4, seq_len_out=8, feat_size_in = 7, feat_size_out = 3, feat_size_hidden = 512,  device=config.device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.feat_size_in = feat_size_in
        self.feat_size_out = feat_size_out
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.seq_len_out_1 = 0
        for i in range(self.seq_len_out):
            self.seq_len_out_1 = self.seq_len_out_1 + i + 1

        encoder_layer = EncoderLayer(d_model=feat_size_hidden, nhead=8, batch_first=True) # 单层多头自注意力
        self.encoder = Encoder(encoder_layer, num_layers=6)                         # transformer编码器

        decoder_layer = DecoderLayer(d_model=feat_size_hidden, nhead=8, batch_first=True) # 单层多头自注意力
        self.decoder = Decoder(decoder_layer, num_layers=6)                         # transformer解码器

        # 用FC代替embedding
        self.embedding_fc_x = nn.Linear(feat_size_in, feat_size_hidden)
        self.embedding_fc_t = nn.Linear(3, feat_size_hidden)

        self.pos_x = position_encode(feat_size_hidden, self.seq_len_in).to(device=device)
        self.pos_t = position_encode(feat_size_hidden, self.seq_len_out).to(device=device)

        self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.seq_len_out).to(device=device)

        self.dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(feat_size_hidden, feat_size_out)

    def forward_standard(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.embedding_fc_x(x)  # 对编码器的输入x进行embedding
        x = x + self.pos_x            # 加上位置码
        x = self.dropout(x)         # dropout
        x = self.encoder(x)         # transformer编码

        if self.training:
            t = self.embedding_fc_t(t)
            t = t + self.pos_t                      # 加上位置码
            t = self.dropout(t)                     # dropout
            t = self.decoder(t, x, self.tgt_mask)   # transformer解码
            t = self.fc(t)
            out = t.reshape(-1, 24)
            out = out.float()
            return out
        else:
            t = torch.zeros(x.size(0), 1, self.feat_size_out, dtype=torch.float).to(x.device)
            for i in range(self.seq_len_out):
                ti = self.embedding_fc_t(t)             # 对解码器的输入t进行embedding
                ti = ti + self.pos_t[:i+1]                # 加上位置码
                ti = self.dropout(ti)                   # dropout
                ti = self.decoder(ti, x)                # transformer解码
                ti = self.fc(ti)                        # 线性层，生成预测点
                t = torch.concat((t,ti[:,i:]), dim=1)   # 将新生成的点加入预测序列
            out = t[:,1:].reshape(-1, 24)
            out =out.float()
            return out
        
    def forward_loop(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_fc_x(x)  # 对编码器的输入x进行embedding
        x = x + self.pos_x            # 加上位置码
        x = self.dropout(x)         # dropout
        x = self.encoder(x)         # transformer编码

        t = torch.zeros(x.size(0), 1, self.feat_size_out, dtype=torch.float).to(x.device)
        for i in range(self.seq_len_out):
            ti = self.embedding_fc_t(t)             # 对解码器的输入t进行embedding
            ti = ti + self.pos_t[:i+1]                # 加上位置码
            ti = self.dropout(ti)                   # dropout
            ti = self.decoder(ti, x)                # transformer解码
            ti = self.fc(ti)                        # 线性层，生成预测点
            t = torch.concat((t,ti[:,i:]), dim=1)   # 将新生成的点加入预测序列
        out = t[:,1:].reshape(-1, 24)
        out =out.float()
        return out
    
    def forward_loop_1(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_fc_x(x)  # 对编码器的输入x进行embedding
        x = x+self.pos_x            # 加上位置码
        x = self.dropout(x)         # dropout
        x = self.encoder(x)         # transformer编码

        t = torch.zeros(x.size(0), 1, self.feat_size_out).to(x.device)
        result = torch.zeros(x.size(0), self.seq_len_out_1, self.feat_size_out).to(x.device)
        s = 0
        for i in range(self.seq_len_out):
            ti = self.embedding_fc_t(t)             # 对解码器的输入t进行embedding
            ti = ti+self.pos_t[:i+1]                # 加上位置码
            ti = self.dropout(ti)                   # dropout
            ti = self.decoder(ti, x)                # transformer解码
            ti = self.fc(ti)                        # 线性层，生成预测点
            t = torch.concat((t,ti[:,i:]), dim=1)   # 将新生成的点加入预测序列
            s = s+i
            result[:,s:s+i+1] = ti
            
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_loop(x)

class DriveUsingCommand(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = "zlg"

        # self.left = MyEgoStateModel.GRU(config.n_feature_dim - config.n_driving_command).to(config.device)
        # self.straight = MyEgoStateModel.GRU(config.n_feature_dim - config.n_driving_command).to(config.device)
        # self.right = MyEgoStateModel.GRU(config.n_feature_dim - config.n_driving_command).to(config.device)
        # self.unknown = MyEgoStateModel.GRU(config.n_feature_dim - config.n_driving_command).to(config.device)

        self.left = TFMModel().to(config.device)
        self.straight = TFMModel().to(config.device)
        self.right = TFMModel().to(config.device)
        self.unknown = TFMModel().to(config.device)

    # def forward(self, x, t, inplace=None):
    #     batch_size, _, _ = x.size()
    #     y = torch.zeros(batch_size, config.n_output_dim * config.n_output_frames, dtype=torch.float).to(config.device)
    #     command = x[:, -1, 7:].view(-1, 4)  # B * 4, the command at current timestamp
    #     command_indices = torch.argmax(command, dim=1).unsqueeze(1)  # B * 1, 选出每一行1的元素对应的index
    #     ego_state = x[:, :, :7]  # ego state B * 4 * 7, 是和上面的command_indices是一一对应的
    #     index0 = torch.where(command_indices == 0)[0]
    #     if len(index0) != 0:
    #         x_left = ego_state[index0, :, :].float()
    #         t_left = t[index0, :, :].float()
    #         x_left = self.left(x_left.to(config.device), t_left.to(config.device))  # B0 * 24
    #         y[index0] = x_left

    #     index1 = torch.where(command_indices == 1)[0]
    #     if len(index1) != 0:
    #         x_straight = ego_state[index1].float()
    #         t_straight = t[index1].float()
    #         x_straight = self.straight(x_straight.to(config.device), t_straight.to(config.device))  # B1 * 24
    #         y[index1] = x_straight

    #     index2 = torch.where(command_indices == 2)[0]
    #     if len(index2) != 0:
    #         x_right = ego_state[index2].float()
    #         t_right = t[index2].float()
    #         x_right = self.right(x_right.to(config.device), t_right.to(config.device))  # B2 * 24
    #         y[index2] = x_right

    #     index3 = torch.where(command_indices == 3)[0]
    #     if len(index3) != 0:
    #         x_unknown = ego_state[index3].float()
    #         t_unknown = t[index3].float()
    #         out_left = self.left(x_unknown.to(config.device), t_unknown.to(config.device))  # B3 * 24
    #         out_straight = self.straight(x_unknown.to(config.device), t_unknown.to(config.device))
    #         out_right = self.right(x_unknown.to(config.device), t_unknown.to(config.device))
    #         x_unknown = (out_left + out_straight + out_right) / 3
    #         y[index3] = x_unknown

    #     y = y.float()

    def forward(self, x, inplace=None):
            batch_size, _, _ = x.size()
            y = torch.zeros(batch_size, config.n_output_dim * config.n_output_frames, dtype=torch.float).to(config.device)
            command = x[:, -1, 7:].view(-1, 4)  # B * 4, the command at current timestamp
            command_indices = torch.argmax(command, dim=1).unsqueeze(1)  # B * 1, 选出每一行1的元素对应的index
            ego_state = x[:, :, :7]  # ego state B * 4 * 7, 是和上面的command_indices是一一对应的
            index0 = torch.where(command_indices == 0)[0]
            if len(index0) != 0:
                x_left = ego_state[index0, :, :].float()
                x_left = self.left(x_left.to(config.device))  # B0 * 24
                y[index0] = x_left

            index1 = torch.where(command_indices == 1)[0]
            if len(index1) != 0:
                x_straight = ego_state[index1].float()
                x_straight = self.straight(x_straight.to(config.device))  # B1 * 24
                y[index1] = x_straight

            index2 = torch.where(command_indices == 2)[0]
            if len(index2) != 0:
                x_right = ego_state[index2].float()
                x_right = self.right(x_right.to(config.device))  # B2 * 24
                y[index2] = x_right

            index3 = torch.where(command_indices == 3)[0]
            if len(index3) != 0:
                x_unknown = ego_state[index3].float()
                out_left = self.left(x_unknown.to(config.device))  # B3 * 24
                out_straight = self.straight(x_unknown.to(config.device))
                out_right = self.right(x_unknown.to(config.device))
                x_unknown = (out_left + out_straight + out_right) / 3
                y[index3] = x_unknown

            y = y.float()
            return y

# 测试 
if __name__ == '__main__':
    device = torch.device('cuda:0')

    model = TFMModel(seq_len_in=4, seq_len_out=8, feat_size_in = 3, feat_size_out = 3, feat_size_hidden = 64, device=device).to(device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    src = torch.rand(2, 4, 3).to(device=device)
    tgt = torch.rand(2, 8, 3).to(device=device)
    y = torch.rand(2, 36, 3).to(device=device)

    loss_f = nn.MSELoss()

    for i in range(10):
        optimizer.zero_grad()
        out = model(src, tgt)
        loss = loss_f(out, y)
        print(loss.item())
        loss.backward()
        optimizer.step()

