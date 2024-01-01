
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5, stride=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 2, stride=3, padding=4)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.sigmoid(v1)
        v4 = v1 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
