
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1, stride=2, padding=2)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv2 = torch.nn.Conv2d(8, 8, 6, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.sigmoid(v1)
        v3 = v2 + 3.
        v4 = self.conv2(v3)
        v5 = v4 + 7.
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
