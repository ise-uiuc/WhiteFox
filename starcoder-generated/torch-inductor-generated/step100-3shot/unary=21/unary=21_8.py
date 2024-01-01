
# Description of model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.tanh(v1)
        return torch.tanh(v2)
# Inputs to the model
x = torch.randn(1, 64, 1, 1)
