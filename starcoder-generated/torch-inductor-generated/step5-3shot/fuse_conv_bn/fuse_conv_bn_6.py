
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x4):
        v4 = self.relu(self.bn(self.c(x4)))
        return v4
# Inputs to the model
x4 = torch.randn(1, 3, 4, 4)
