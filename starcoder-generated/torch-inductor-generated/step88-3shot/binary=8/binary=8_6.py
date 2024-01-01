
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1, x2):
        v1 = self.conv1(x2)
        v2 = self.conv2(x2)
        v3 = torch.add(x1, v1)
        v4 = torch.add(v3, v2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
