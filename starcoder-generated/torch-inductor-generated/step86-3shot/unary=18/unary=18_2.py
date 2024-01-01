
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(3, 2, kernel_size=3, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v4 = self.conv1(v2)
        v5 = torch.sigmoid(v4)
        v6 = self.conv2(x2)
        v7 = torch.sigmoid(v6)
        return v5 + v7
# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)
x2 = torch.randn(1, 3, 10, 10)
