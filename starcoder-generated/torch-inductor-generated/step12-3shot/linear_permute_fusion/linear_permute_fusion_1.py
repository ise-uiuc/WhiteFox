
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 10, (5, 5), stride=(3, 3))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sum(v1, dim=2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 3, 3)
