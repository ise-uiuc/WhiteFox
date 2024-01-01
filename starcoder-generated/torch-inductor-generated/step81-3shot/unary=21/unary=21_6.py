
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
tensor = torch.randn(1, 1, 1, 1)
