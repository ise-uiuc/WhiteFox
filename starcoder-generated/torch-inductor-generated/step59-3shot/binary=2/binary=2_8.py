
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x) * 5
        t = torch.randn(1)
        return v1 ** t
# Inputs to the model
x = torch.randn(1, 3, 256, 256)
