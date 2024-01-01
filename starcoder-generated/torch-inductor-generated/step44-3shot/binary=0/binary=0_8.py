
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 5, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(5, 5, 3, stride=1, padding=1)
    def forward(self, x1, other=3, padding1=3):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        if padding1 == 3:
            padding1 = torch.randn(v1.shape)
        v3 = v1 + other
        v4 = torch.cat([v3, padding1]) + v2
        return v4
# Inputs to the model
x1 = torch.randn(2, 5, 64, 64)
