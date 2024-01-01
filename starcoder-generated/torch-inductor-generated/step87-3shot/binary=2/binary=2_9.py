
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 9, 7)
    def forward(self, x6):
        v1 = self.conv(x6)
        v2 = v1 - 4.0
        return v2
# Inputs to the model
x6 = torch.randn(1, 2, 20, 20)
