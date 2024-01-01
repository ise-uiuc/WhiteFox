
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = torch.tanh(self.conv1(x1))
        v2 = torch.sigmoid(self.conv1(x1))
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
