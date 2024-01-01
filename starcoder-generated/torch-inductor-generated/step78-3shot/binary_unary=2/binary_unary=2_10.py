
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 40, 1, stride=1, padding=0)
        self.mul = torch.mul
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.mul(v1, 7.5)
        v3 = F.relu(v2)
        v4 = torch.squeeze(v3, 0)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 56, 56)
