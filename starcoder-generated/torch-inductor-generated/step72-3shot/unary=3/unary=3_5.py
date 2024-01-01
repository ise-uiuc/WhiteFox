
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 1 / 0.8325546111576977
        v3 = v1 * v2
        v4 = self.conv2(v3)
        v5 = 1 / 2.718281752767521
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
