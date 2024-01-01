
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = torch.transpose(v1, 1, 0)
        v2 = torch.transpose(v1, 2, 3)
        return torch.dot(v1, v2)
# Inputs to the model
x1 = torch.randn(32, 3, 64, 64)
