
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=False):
        v1 = self.conv2(self.conv1(x1))
        v2 = self.conv3(v1)
        v3 = v2 + 0.1
        v4 = v3 + 0.1
        v5 = v4 + 0.1
        v6 = v5 + 0.1
        if other == False:
            other = torch.randn(v6.shape)
        v7 = v6 + other
        return v7
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
