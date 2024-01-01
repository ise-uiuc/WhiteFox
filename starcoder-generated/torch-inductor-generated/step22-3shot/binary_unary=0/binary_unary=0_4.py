
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        return torch.tanh(v3) + x2
# Inputs to the model
x1 = torch.randn(1, 16, 16, 16)
x4 = torch.randn(1, 16, 16, 16)
