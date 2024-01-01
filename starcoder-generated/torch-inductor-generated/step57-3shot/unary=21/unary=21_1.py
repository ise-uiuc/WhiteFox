
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v1 = torch.tanh(v1)
        v2 = torch.tanh(v1)
        v2 = torch.tanh(v2)
        return v2
# Inputs to the model
x = torch.randn(1, 32, 224, 224)
