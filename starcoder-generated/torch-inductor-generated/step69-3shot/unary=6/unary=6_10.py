
class Model(torch.nn.Module):
    def __init__(self, ch1, p1):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, ch1, p1, stride=1, padding=1)
    def forward(self, x1):
        p2 = 0
        t1 = 6.0
        v1 = self.conv(x1)
        v2 = v1 + p2
        v3 = torch.nn.functional.relu(v2)
        v4 = v3 * p2
        v5 = v4 / t1
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
