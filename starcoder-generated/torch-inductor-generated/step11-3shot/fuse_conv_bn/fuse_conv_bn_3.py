
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(1, 4, 3)
        self.bn1 = torch.nn.BatchNorm2d(4)
    def forward(self, x):
        x = self.conv3(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x.sum()
# Inputs to the model
x1 = torch.randn(1,1,1,1)
