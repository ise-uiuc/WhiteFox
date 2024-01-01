
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 1, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        return v2.sum(dim=0)
# Inputs to the model
x1 = torch.randn(2, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
x3 = torch.randn(2, 3, 32, 32)
