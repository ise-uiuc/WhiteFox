
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 25, 1, stride=4, padding=6)
    def forward(self, x):
        v1 = torch.tanh(x)
        self.conv(v1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(128, 16, 16, 16)
