
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x1, other=True):
        v1 = self.conv(x1)
        v2 = self.linear(v1)
        if other == True:
            other = torch.randn(v1.shape)
        v4 = v2 + other
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
