
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 4, 1, stride=1, padding=1)
    def forward(self, x1, other=1):
        v1 = self.conv1(x1)
        if other == 1:
            other = torch.randn(v1.shape)
        v2 = self.conv2(x1)
        if other == 1:
            other = torch.randn(v2.shape)
        return (v1 + other) + (v2 + other)
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
