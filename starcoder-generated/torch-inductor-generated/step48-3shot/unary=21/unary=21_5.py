
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 1, 1, padding=0)
        self.conv1 = torch.nn.Conv2d(1, 7, 3, 1, padding=1)
        self.conv2 = torch.nn.Conv2d(7, 1, 1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        v3 = self.conv1(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv2(v4)
        return v5
# Inputs to the model
x = torch.randn(71, 7, 71, 70)
