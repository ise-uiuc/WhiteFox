
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 5, stride=3, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 2.0
        v3 = torch.squeeze(v2, 0)
        v4 = F.relu(v3)
        v5 = torch.mean(v1, dim = 0)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 56, 56)
