
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 96, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(96, 54, 1, stride=1, padding=0)
    def forward(self, x1, other=None):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        if other == None:
            other = torch.randn(v2.shape)
        v3 = v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
