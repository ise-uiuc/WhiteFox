
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(16, 16, 1)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(16)
        self.bn = nn.BatchNorm2d(4)
    def forward(self, x):
        x = nn.functional.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 16, 32, 32)
