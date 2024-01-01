
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 128, 1, stride=1)
    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        out = x
        return out
# Inputs to the model
x = torch.randn(1, 128, 64, 64)
