
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
    def forward(self, x1, x2):
        v1 = self.conv1(torch.tanh(x1))
        v2 = self.conv2(x2)
        v3 = v2 + torch.sigmoid(v1)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x1 = torch.randn(16, 8, 64, 64)
x2 = torch.randn(16, 8, 64, 64)
