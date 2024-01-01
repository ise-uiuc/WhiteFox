
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.bn0 = torch.nn.BatchNorm2d(3)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn0(x)
        x = self.bn1(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
