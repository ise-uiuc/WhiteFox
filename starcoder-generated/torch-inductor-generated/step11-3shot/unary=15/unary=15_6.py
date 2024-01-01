
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        a, b, c = 3, 16, 3
        self.conv1 = torch.nn.Conv2d(a, b, c)
        self.conv2 = torch.nn.Conv2d(b, c, a)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
