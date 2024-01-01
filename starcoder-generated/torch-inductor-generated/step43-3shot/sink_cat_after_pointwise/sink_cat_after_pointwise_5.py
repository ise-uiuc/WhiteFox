
import random
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand = random.randint(0, 1)
    def forward(self, x):
        def concat_in_dim(x):
            return [torch.cat(x, dim=1), torch.cat(x, dim=1)]
        def concat_in_dim2(x):
            return [torch.cat(x, dim=2), torch.cat(x, dim=2)]
        def concat_in_dim3(x):
            return [torch.cat(x, dim=3), torch.cat(x, dim=3)]
        x = concat_in_dim([x, x, x]) 
        x = concat_in_dim(x) if self.rand else x
        x = concat_in_dim(x) if self.rand else x
        x = concat_in_dim(x) if self.rand else x
        x = concat_in_dim2(x) if self.rand else x
        x = concat_in_dim3(x) if self.rand else x
        x = torch.relu(x)
        x = x[0] 
        return x
# Inputs to the model
x = torch.randn(2, 3, 4, 5, 6)
