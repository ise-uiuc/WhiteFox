
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 15, 52, stride=3, padding=22)
    def forward(self, x2):
        v1 = self.conv(x2)
        return v1
# Inputs to the model
x2 = torch.randn(1, 1, 56, 40)
