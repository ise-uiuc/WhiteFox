
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential()
        self.seq.add_module('conv', torch.nn.ConvTranspose2d(3, 32, 3, padding=1, stride=2))
    def forward(self, x1):
        v1 = self.seq(x1)
        v2 = v1.permute(0, 3, 2, 1)
        v3 = torch.relu(v1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
