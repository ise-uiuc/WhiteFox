
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 1)
        self.linear1 = torch.nn.Linear(192, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.sigmoid()
        v3 = v1 * v2
        v4 = v3.flatten(1)
        v5 = self.linear1(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
