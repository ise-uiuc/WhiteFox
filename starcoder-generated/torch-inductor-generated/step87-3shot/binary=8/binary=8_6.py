
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(torch.add(x1, x2))
        v2 = self.conv2(torch.add(x1, x2))
        v3 = v1 + v2
        v4 = v3.add(v2)
        v5 = v3.add(v3)
        v6 = v3 + torch.add(v1, v2)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
