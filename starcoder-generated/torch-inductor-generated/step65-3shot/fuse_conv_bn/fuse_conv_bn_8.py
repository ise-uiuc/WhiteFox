
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.layer = torch.nn.Sequential(F.conv3d(4, 5, 3, bias=True), F.batch_norm(5), F.relu6())
    def forward(self, x1):
        s1 = self.layer(x1)
        return s1 + s1
# Inputs to the model
x1 = torch.randn(2, 4, 4, 4, 4)
