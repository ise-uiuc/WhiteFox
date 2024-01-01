
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, 2, stride=1, padding=0)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = v1 - 5.1
        return v2
# Inputs to the model
x2 = torch.randn(1, 2, 64, 64)
