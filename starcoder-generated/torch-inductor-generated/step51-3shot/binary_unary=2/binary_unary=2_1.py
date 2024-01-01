
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 49, padding=24, groups=1)
        self.conv2 = torch.nn.Conv2d(4, 1, 7, padding=3, groups=1)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = F.relu(x2)
        x4 = self.conv2(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
