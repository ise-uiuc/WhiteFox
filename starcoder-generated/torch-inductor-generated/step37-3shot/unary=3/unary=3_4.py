
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 8, 1, stride=1, padding=0)
        self.batch_norm = torch.nn.BatchNorm1d((56, 56))
        self.conv2 = torch.nn.Conv2d(8, 4, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 64, 56, 56)
