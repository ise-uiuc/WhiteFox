
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(512, 64, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 / 2
        v3 = torch.tanh(v2)
        v4 = v3.relu()
        v5 = v4.hardtanh(max_val=3)
        v6 = v5.tanh()
        v7 = v6.sigmoid()
        v8 = v7 * v1
        return v8
# Inputs to the model
x1 = torch.randn(1, 512, 64, 64)
