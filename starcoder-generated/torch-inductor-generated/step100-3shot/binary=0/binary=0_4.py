
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 4, 1)
        self.conv2 = torch.nn.Conv2d(3, 4, 1)
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, other=None):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        x3 = v1 + v2
        x4 = self.conv1(x3)
        x5 = self.conv1(x1)
        if other == None:
            other = torch.randn(x1.shape).to(x1.dtype).to(x1.device)
        x6 = x5 + other
        x7 = self.conv1(x1)
        x8 = self.conv2(x2)
        x9 = self.conv2(x1)
        return (x3, x4, x6, x7, x8, x9)
# Inputs to the model
x1 = torch.randn(1, 2, 1, 1)
x2 = torch.randn(1, 3, 16, 16)
x3 = torch.randn(1, 2, 16, 16)
x4 = torch.randn(1, 2, 8, 8)
x5 = torch.randn(1, 2, 8, 8)
x6 = torch.randn(1, 2, 32, 32)
x7 = torch.randn(1, 2, 100, 100)
x8 = torch.randn(1, 3, 20, 20)
x9 = torch.randn(1, 3, 1, 1)
