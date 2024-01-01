
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convBN = F.conv3d(7, 3, 3)
    def forward(self, x1):
        s = F.relu(self.convBN(x1))
        return s
# Inputs to the model
x1 = torch.randn(1, 7, 4, 4, 4)
