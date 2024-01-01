
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
