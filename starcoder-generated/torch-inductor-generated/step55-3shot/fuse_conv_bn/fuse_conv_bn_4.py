
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 256, (3, 3), stride=(2, 2), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(256)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return x
# Inputs to the model
x = torch.randn(1, 1, 5, 5)
