
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(22, 96, 2, stride=1)
        self.conv1 = torch.nn.Conv2d(96, 96, 2, stride=1)
    def forward(self, x4):
        v3 = self.conv(x4)
        v3 = torch.tanh(v3)
        return v3
# Inputs to the model
x4 = torch.randn(1, 22, 33, 33)
