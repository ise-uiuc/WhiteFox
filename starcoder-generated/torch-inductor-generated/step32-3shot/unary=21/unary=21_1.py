
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 1, 1, 1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x = torch.randn(64, 3, 28, 28)
