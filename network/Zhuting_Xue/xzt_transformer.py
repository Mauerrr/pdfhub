import torch
import torch.nn as nn
import torchvision
import numpy as np
from einops import rearrange
import train.Zhuting_Xue.train_config as config

def gen_pose_mask(N):
    mask = torch.triu(torch.ones(N,N),diagonal=1).bool()
    # "diagonal = 0" means all elements on and above the main diagonal are retained
    """
    tensor([[1., 1., 1., 1., 1.],
            [0., 1., 1., 1., 1.],
            [0., 0., 1., 1., 1.],
            [0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 1.]])
    """
    return mask

class FeedForward(nn.Module):
    def __init__(self, dim = config.dim, hidden_dim = config.hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim, bias=False))
        
    def forward(self, x):                            
        return self.fc(x)

class PositionalEncoding(nn.Module):
    def __init__(self, dim = config.dim, dropout=0.1, max_len=5000): # max_len: max length of the input sequence
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) 
        pos_table = np.array([
        [pos / np.power(10000, 2 * i / dim) for i in range(dim)]
        if pos != 0 else np.zeros(dim) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])                  
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])                  
        self.pos_table = torch.FloatTensor(pos_table).cuda() # (max_len, dim)           

    def forward(self, x):                                      
        return self.pos_table[:x.size(1), :]

class SelfAttention(nn.Module):
    def __init__(self, dim = config.dim, heads = config.heads, dim_head = config.dim_head):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, mask):
        x = self.norm(x)
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv) # 对q，k，v分别执行：dim进行reshape成h * d

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # (b, h, n, d) * (b, h, d, n) = (b, h, n, n)

        if mask is not None:
            min_neg_value = -torch.finfo(dots.dtype).max
            mask = mask.unsqueeze(1).repeat(1,h,1,1)
            dots.masked_fill_(mask, min_neg_value)
        
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, heads = config.heads, dim_head = config.dim_head):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(q_dim)
        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(q_dim, inner_dim, bias = False)
        self.to_k = nn.Linear(k_dim, inner_dim, bias = False)
        self.to_v = nn.Linear(v_dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, q_dim, bias = False)

    def forward(self, x, key, value, mask=None):
        x = self.norm(x)
        h = self.heads
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h = h)
        k = rearrange(self.to_k(key), 'b n (h d) -> b h n d', h = h)
        v = rearrange(self.to_v(value), 'b n (h d) -> b h n d', h = h)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            min_neg_value = -torch.finfo(dots.dtype).max
            mask = mask.unsqueeze(1).repeat(1,h,1,1)
            dots.masked_fill_(mask, min_neg_value)
        
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class EncoderBlock(nn.Module):
    def __init__(self, dim = config.dim , heads = config.heads , dim_head = config.dim_head , hidden_dim = config):
        super().__init__()
        self.attn = SelfAttention(dim, heads = heads, dim_head = dim_head)
        self.ff = FeedForward(dim, hidden_dim)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
    
    def forward(self, x, mask):
        x = self.norm_1(self.attn(x, mask)) + x
        x = self.norm_2(self.ff(x)) + x
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim = config.dim , depth = config.enc_depth, heads = config.heads, dim_head = config.dim_head, hidden_dim = config.hidden_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(EncoderBlock(dim, heads, dim_head, hidden_dim))
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, dim = config.dim, heads = config.heads, dim_head = config.dim_head, hidden_dim = config.hidden_dim):
        super().__init__()
        self.attn_1 = SelfAttention(dim, heads, dim_head)
        self.attn_2 = CrossAttention(dim, dim, dim, heads, dim_head)
        self.ff = FeedForward(dim, hidden_dim)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.norm_3 = nn.LayerNorm(dim)
    
    def forward(self, x, enc_out, mask):
        x = self.norm_1(self.attn_1(x, mask)) + x
        x = self.norm_2(self.attn_2(x, enc_out, enc_out, mask=None)) + x
        x = self.norm_3(self.ff(x)) + x
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, dim = config.dim, depth = config.dec_depth, heads = config.heads, dim_head = config.dim_head, hidden_dim = config.hidden_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(DecoderBlock(dim, heads, dim_head, hidden_dim))
        
    def forward(self, x, enc_out, mask):                               
        for layer in self.layers:
            x = layer(x, enc_out, mask)
        return x

class Transformer(nn.Module):
    def __init__(self, dim = config.dim, enc_depth = config.enc_depth, dec_depth = config.dec_depth, heads = config.heads, dim_head = config.dim_head, hidden_dim = config.hidden_dim, input_n=config.n_input_frames, predict_n=config.n_output_frames, training=True):
        super().__init__()
        self.training = training
        self.predict_n = predict_n
        # encode image
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Identity()
        self.model_type = "xzt_transformer"
        
        self.embeder_img = nn.Sequential(
            self.resnet50,
            nn.Linear(2048, dim),
            nn.LayerNorm(dim)
        )
        
        self.embeder_coord = nn.Sequential(
            nn.Linear(3, dim),
            nn.LayerNorm(dim)
        )

        self.encoder_img = TransformerEncoder(dim, enc_depth, heads, dim_head, hidden_dim)
        self.encoder_coord = TransformerEncoder(dim, enc_depth, heads, dim_head, hidden_dim)
        self.decoder = TransformerDecoder(dim, dec_depth, heads, dim_head, hidden_dim)
        self.pos_encoding_img = PositionalEncoding(dim, max_len=max(input_n, predict_n))
        self.pos_encoding_coord = PositionalEncoding(dim, max_len=max(input_n, predict_n))
        self.final_fc = nn.Linear(dim, 3)
    
    def forward(self, x=None, coord=None, target=None):
        # x: (B, 5, 3, H, W)
        # coord: (B, 5, 3)
        # prepare input

        N = coord.shape[0] # batch size
        
        """
        x = x.flatten(0, 1)
        x = self.embeder_img(x)
        x = x.view(N, -1, x.shape[-1]) # (B, 5, 256)
        x += self.pos_encoding_img(x).unsqueeze(0).repeat(N, 1, 1) # pos_encoding: (5, 256) -> (B, 5, 256) 
        """
        
        coord = coord.flatten(0, 1)
        coord = self.embeder_coord(coord) 
        coord = coord.view(N, -1, coord.shape[-1])
        coord += self.pos_encoding_coord(coord).unsqueeze(0).repeat(N, 1, 1)
        
        # encode image and coord
        
        # enc_out_img = self.encoder_img(x, mask=None)
        
        enc_out_coord = self.encoder_coord(coord, mask=None)
        
        # combine image and coord
        # naive addition
        # enc_out = enc_out_img + enc_out_coord

        # naive multiplication
        # enc_out = enc_out_img * enc_out_coord

        # naive concatenation
        # enc_out = torch.cat((enc_out_img, enc_out_coord), dim=1)
        
        # when no image input
        enc_out = enc_out_coord    

        # prepare decoder input
        start = torch.zeros(N, 1, 3).cuda()
        """
        # use GT to train
        if self.training:
            target = torch.cat((start, target), dim=1)[:,:-1,:] # discard the last coordinate -> (N, 6, 3)
            target = target.flatten(0, 1)
            target = self.embeder_coord(target) 
            target = target.view(N, -1, target.shape[-1])
            target += self.pos_encoding_coord(target).unsqueeze(0).repeat(N, 1, 1)
            # generate mask
            mask = gen_pose_mask(self.predict_n).unsqueeze(0).repeat(N, 1, 1).cuda()
            dec_out = self.decoder(target, enc_out, mask)
            return self.final_fc(dec_out)
        else:
            # variant length sequence input
            preds = start
            for i in range(self.predict_n):
                preds_i = preds.flatten(0, 1)
                preds_i = self.embeder_coord(preds_i) 
                preds_i = preds_i.view(N, -1, preds_i.shape[-1])
                preds_i += self.pos_encoding_coord(preds_i).unsqueeze(0).repeat(N, 1, 1)
                dec_out = self.decoder(preds_i, enc_out, mask=None)
                
                new_preds = self.final_fc(dec_out)
                preds = torch.cat((preds, new_preds[:,-1:]), dim=1)
            return preds[:, 1:]
        """

        # use prediction to train -> identical training and inference
        # variant length sequence input
        preds = start
        for i in range(self.predict_n):
            preds_i = preds.flatten(0, 1)
            preds_i = self.embeder_coord(preds_i) 
            preds_i = preds_i.view(N, -1, preds_i.shape[-1])
            preds_i += self.pos_encoding_coord(preds_i).unsqueeze(0).repeat(N, 1, 1)
            dec_out = self.decoder(preds_i, enc_out, mask=None)
            
            new_preds = self.final_fc(dec_out)
            preds = torch.cat((preds, new_preds[:,-1:]), dim=1)
        return preds[:, 1:]

    
