
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(8, 8, 1)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        x = nn.functional.relu(x)
        x = nn.functional.relu(x)
        x = nn.functional.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 8, 32, 32)
