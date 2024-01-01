
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(14, 1, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(14, 1, 1, stride=1, padding=1)
    def forward(self, x, other=None):
        v1 = self.conv1(x)
        v2 = self.conv3(x)
        if other.is_cuda:
            v3 = (v1 - v2) + other.to(dtype=v1.dtype)
        else:
            v3 = (v1 - v2) + other
        return v3
# Inputs to the model
x = torch.randn(1, 14, 64, 64)
