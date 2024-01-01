
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, 2)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(2)
    def forward(self, x):
        x1 = torch.nn.functional.relu(self.conv1(x))
        y1 = self.bn(x1)
        x2 = torch.nn.functional.relu(self.conv1(y1))
        y2 = self.bn(x2)
        return y2
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
