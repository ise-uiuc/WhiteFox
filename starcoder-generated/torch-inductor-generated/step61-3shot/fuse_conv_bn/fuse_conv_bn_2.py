
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32, affine=False)
        self.bn2 = torch.nn.BatchNorm2d(32)
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.bn1(self.conv2(x))
        return self.bn2(x)
# Inputs to the model
x = torch.randn(20, 64, 10, 10)
