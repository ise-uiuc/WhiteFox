
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = torch.nn.Conv2d(4, 4, 1)
        torch.manual_seed(1)
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.conv2 = torch.nn.Conv2d(4, 4, 1)
        torch.manual_seed(1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.randn(1, 4, 32, 32)
