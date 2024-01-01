
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = 0.9762326874978142 * v1 * v2
        v4 = v3.matmul(v3)
        v5 = 0.3884185366798307 * v3
        return v4
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
