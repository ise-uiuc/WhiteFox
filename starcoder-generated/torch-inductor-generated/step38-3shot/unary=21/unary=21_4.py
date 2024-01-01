
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(18, 4, 3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(3, stride=3)
        self.conv2 = torch.nn.Conv2d(12, 1, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = torch.tanh(self.conv(x))
        return torch.tanh(self.pool(v1))
# Inputs to the model
x = torch.randn(1, 18, 64, 64)
