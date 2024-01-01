
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = torch.nn.functional.relu(x)
        x = x + 1
        return x
# Inputs to the model
x = torch.randn(1, 3, 3, 3)
