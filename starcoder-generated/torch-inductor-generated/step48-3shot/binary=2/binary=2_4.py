
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 8, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1)
    def forward(self, x2):
        v1 = torch.tanh(self.conv1(x2))
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x2 = torch.randn(1, 64, 64, 64)
