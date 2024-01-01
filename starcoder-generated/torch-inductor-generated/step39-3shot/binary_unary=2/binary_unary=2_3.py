
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, (6, 7), stride=(2, 3), padding=(2, 3))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 1
        _1 = torch.sqrt(torch.abs(v2))
        v3 = F.relu(_1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 192, 256)
