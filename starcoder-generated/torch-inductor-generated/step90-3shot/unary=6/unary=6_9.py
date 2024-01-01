
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = 0 + v2
        v4 = 6.0 < v3
        v5 = v1 * v4
        v6 = v5 / 6
        return v6.unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)   
