
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 5, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - torch.tensor(2, dtype=torch.float)
        return v2
# Inputs to the model
x = torch.randn(1, 2, 10, 10)
