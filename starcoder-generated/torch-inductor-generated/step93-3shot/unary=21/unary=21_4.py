
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.conv = torch.nn.Conv2d(64, 64, 3, padding=1, stride=1)
    def forward(self, x):
        v1 = self.pool(x)
        v2 = self.conv(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 64, 57, 57)
