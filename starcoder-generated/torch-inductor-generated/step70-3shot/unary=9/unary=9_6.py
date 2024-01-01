
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        v1 = t1 / 5 + t.tanh(t1)
        v2 = v1 - 5
        t2 = torch.nn.functional.sigmoid(v2)
        v3 = torch.clamp_max(t2, 6)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
