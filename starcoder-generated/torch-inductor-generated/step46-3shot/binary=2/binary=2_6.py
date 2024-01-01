
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - torch.tensor([1.5231, 1.0045, 1.0433])
        return v2
# Inputs to the model
x = torch.randn(1, 3, 2, 2)
