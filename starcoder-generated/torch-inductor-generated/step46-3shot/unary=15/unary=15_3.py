
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 12, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(12, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        return torch.sigmoid(v4)
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
