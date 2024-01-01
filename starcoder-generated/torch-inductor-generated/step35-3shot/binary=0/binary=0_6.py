
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 64, 1, stride=1, padding=5, bias=True)
        self.conv2 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=5, bias=True)
    def forward(self, x1, other=None):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        if other == None:
            other = torch.randn(v2.shape)
        v3 = v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(1, 6, 8, 8)
