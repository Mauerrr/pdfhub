import torch
from torch import Tensor
from torch import nn

class MLPModel(nn.Module):
    def __init__(self, pretrained=True, input_n=4, out_features=16) -> None:
        super().__init__()
        self.out_features = out_features
        self.mlp0 = nn.Sequential(nn.Linear(in_features=2*input_n-2, out_features=512),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(in_features=512, out_features=256),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(in_features=256, out_features=out_features)
                                  )
        self.mlp1 = nn.Sequential(nn.Linear(in_features=2*input_n-2, out_features=512),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(in_features=512, out_features=256),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(in_features=256, out_features=out_features)
                                  )
        self.mlp2 = nn.Sequential(nn.Linear(in_features=2*input_n-2, out_features=512),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(in_features=512, out_features=256),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(in_features=256, out_features=out_features)
                                  )
        self.mlp3 = nn.Sequential(nn.Linear(in_features=2*input_n-2, out_features=512),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(in_features=512, out_features=256),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(in_features=256, out_features=out_features)
                                  )


    def forward_dc(self, x: Tensor, dc: Tensor) -> Tensor:
        
        i0 = torch.where(dc == 0)
        i1 = torch.where(dc == 1)
        i2 = torch.where(dc == 2)
        i3 = torch.where(dc == 3)

        i0_l = len(i0[0])
        i1_l = len(i1[0])
        i2_l = len(i2[0])
        i3_l = len(i3[0])

        y = torch.zeros((x.size(0),self.out_features)).to(x.device)
        if i0_l > 0:
            y[i0] = self.mlp0(x[i0])
        if i1_l > 0:
            y[i1] = self.mlp1(x[i1])
        if i2_l > 0:
            y[i2] = self.mlp2(x[i2])
        if i3_l > 0:
            y[i3] = self.mlp3(x[i3])
        
        return y
    
    def forward(self, x: Tensor, dc: Tensor) -> Tensor:
        x = self.mlp0(x)
        return x
