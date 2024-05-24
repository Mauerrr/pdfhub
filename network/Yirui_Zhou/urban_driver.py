import torch
import data_prep.Yirui_Zhou.config as config
import torchvision.models as models
import torch.nn as nn
import math
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class Resnet_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self._init_resnet_model()
    def forward(self, images):
        images = images
        images = images.reshape(-1, 3, config.image_height, config.image_width)
        return self.resnet_model(images)

    def resnet_model(self, x):
        x = self.resnet50_model(x)
        x = self.resnet50_model_final_fc(x)
        batch_size = x.size(0) // config.n_input_image_frames
        return x.view(batch_size, -1, config.embedding_dim)

    def _init_resnet_model(self):
        self.resnet50_model = models.resnet50(pretrained=True)
        self.resnet50_model.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.resnet50_model_final_fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, config.embedding_dim)
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = config.embedding_dim, dropout: float = 0.1, max_length: int = 5000):
        """
        Args:
          d_model:      dimension of embeddings
          dropout:      randomly zeroes-out some of the input
          max_length:   max sequence length
        """
        super(PositionalEncoding, self).__init__()
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
        # calc cosine on odd indices
        pe[:, 1::2] = torch.cos(k * div_term)
        # add dimension
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
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        # perform dropout
        return self.dropout(x)

class Target_embedding(nn.Module):
    """
    target embedding
    """
    def __init__(self, emb_size: int = config.embedding_dim*config.n_output_frames):
        super(Target_embedding, self).__init__()
        self.embedding = nn.Linear(config.n_output_frames * config.trajectory_xy_dim, emb_size)

    def forward(self, tgt: Tensor):
        target_embedding = self.embedding(tgt)
        target_embedding = target_embedding.view(-1, config.n_output_frames, config.embedding_dim)
        return target_embedding

class TransformerModel(nn.Module):
    def __init__(self,
                 d_model: int = config.embedding_dim,
                 nhead: int = 8,
                 nlayers: int = 1,
                 dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.tgt_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout, 5000)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layers, nlayers)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.decoder = TransformerDecoder(decoder_layers, nlayers)
        self.backbone = Resnet_backbone()
        self.tgt_embedding = Target_embedding(d_model * config.n_output_frames)

    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square causal mask for the sequence.
        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.get_default_dtype()
        return torch.triu(torch.full((sz, sz), float('-inf'), dtype=dtype, device=device), diagonal=1)

    def forward(self, images, tgt):
        src = self.backbone(images) # [batch_size, n_input_frames, embedding_dim]
        src = self.pos_encoder(src) # [batch_size, n_input_frames, embedding_dim]
        tgt_len = tgt.size()[1]
        tgt = tgt.view(-1, config.n_output_frames*config.trajectory_xy_dim) # [batch_size, n_output_frames*2]
        tgt = self.tgt_embedding(tgt) # [batch_size, n_output_frames, embedding_dim]
        tgt = self.pos_encoder(tgt) # [batch_size, n_output_frames, embedding_dim]
        if self.tgt_mask is None:
            device = tgt.device
            mask = self._generate_square_subsequent_mask(tgt_len).to(device)
            self.tgt_mask = mask
        memory = self.encoder(src)
        output = self.decoder(tgt, memory, tgt_mask=self.tgt_mask)
        output = output.view(-1, config.embedding_dim*config.n_output_frames)# [batch_size, n_output_frames*embedding_dim]
        output = nn.Linear(config.embedding_dim * config.n_output_frames, config.trajectory_xy_dim * config.n_output_frames)(output)
        output = output.view(-1, config.n_output_frames, config.trajectory_xy_dim) # [batch_size, n_output_frames, 2]
        return output







