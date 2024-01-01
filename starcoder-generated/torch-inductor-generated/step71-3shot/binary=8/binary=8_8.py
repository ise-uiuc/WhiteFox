
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, 1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = v3 + 1
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 1024, 1024)
x2 = torch.randn(1, 3, 1024, 1024)
