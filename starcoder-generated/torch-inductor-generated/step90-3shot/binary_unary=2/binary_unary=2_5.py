
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=3, padding=2, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.01
        v3 = F.relu(v2)
        v4 = torch.cat((v3, v3), 1)
        _1, _2, _3, v5 = v4.size()
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
