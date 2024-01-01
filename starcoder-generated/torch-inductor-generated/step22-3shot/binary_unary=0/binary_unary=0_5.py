
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.maxpool2d(v1, 3, stride=1, padding=1)
        v3 = v2 + v1
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
