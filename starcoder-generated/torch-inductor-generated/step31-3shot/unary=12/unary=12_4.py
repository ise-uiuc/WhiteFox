
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(3, 8, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.relu(x1)
        v2 = self.conv(v1)
        v3 = v2.sum(dim=-1).sum(dim=-1).expand_as(v1)
        v4 = self.relu(v1)
        v5 = self.conv(v4)
        v6 = v5 * v3
        v7 = v6.sum(dim=-1).sum(dim=-1).expand_as(v1)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
