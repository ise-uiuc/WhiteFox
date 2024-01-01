
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1, x2):
        x3 = torch.add(x1, x2)
        x4 = self.conv1(x3)
        return x4
# Inputs to the model
x1 = torch.randn(2, 3, 16, 16)
x2 = torch.randn(2, 3, 16, 16)
