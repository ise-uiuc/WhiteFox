
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1)
    def forward(self, x1):
        x2 = F.pad(x1, (1, 2, 1, 2))
        x3 = self.conv1(x2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 3, 6, 5)
