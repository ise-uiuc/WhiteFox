
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, (1, 4), stride=(1, 1), padding=(0, 1))
        self.conv2 = torch.nn.Conv2d(1, 1, (1, 3), stride=(1, 1), padding=(0, 1))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(x1)
        v3 = torch.erf(v1)
        v4 = torch.erf(v2)
        v5 = v3 + v4
        v6 = v2 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 45, 20)
