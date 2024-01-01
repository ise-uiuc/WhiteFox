
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 6, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - torch.randn([(1),{2}])
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 1, 1)
