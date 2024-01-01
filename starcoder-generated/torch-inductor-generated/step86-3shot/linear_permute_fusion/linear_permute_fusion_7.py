
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, 1, 0, 2)
        self.linear = torch.nn.Linear(3, 1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1.permute(0, 1, 2, 3)
        v3 = self.linear(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 1, 3, 3)
