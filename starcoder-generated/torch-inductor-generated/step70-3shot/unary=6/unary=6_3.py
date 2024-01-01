
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 7, 3, stride=1, padding=1)
    def forward(self, x1):
        n1 = self.conv(x1)
        n2 = n1 - 3
        n3 = torch.clamp(n2, 4, 10)
        n4 = n1 / n3
        n5 = torch.sin(n4)
        return n5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
