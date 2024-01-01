
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 4, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(4)
    def forward(self, x3):
        return self.bn2(self.conv2(self.conv1(x3)))
# Inputs to the model
x3 = torch.randn(1, 4, 4, 4)
