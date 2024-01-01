
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.cat([(v2[0, 0, 0, 0]).unsqueeze(-1), (v2[0, 0, 1, 0]).unsqueeze(-1)], 0)
        v4 = -v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 8, 8)
