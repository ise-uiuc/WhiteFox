
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = torch.mul(self.conv(x1), 6)
        t2 = torch.add(t1, 3)
        output = torch.clamp(t2, 0, 6)
        return output
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
