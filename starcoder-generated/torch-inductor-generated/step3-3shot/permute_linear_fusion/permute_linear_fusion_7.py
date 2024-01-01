
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 5, 3, groups = 5)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = self.conv(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 5, 20, 20)
