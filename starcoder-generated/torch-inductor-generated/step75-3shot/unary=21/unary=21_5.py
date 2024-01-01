
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 27, 1, stride=3, padding=1)
    def forward(self, x4):
        v1 = self.conv(x4)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x4 = torch.randn(1, 3, 13, 13)
