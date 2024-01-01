
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if other is None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        v2 = self.linear(v2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
