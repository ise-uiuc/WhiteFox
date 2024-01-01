
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 35, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(78, 3, 1, stride=1, padding=1)
    def forward(self, x1, other=0):
        v1 = self.conv1(x1)
        if other == 0:
            other = torch.randn(v1.shape)
        v2 = self.conv2(v1 + other)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
