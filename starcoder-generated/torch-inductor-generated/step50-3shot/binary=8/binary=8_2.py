
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1.reshape(v1.size(0), v1.size(1) * v1.size(2) * v1.size(3))
        v4 = v3 + v2
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
