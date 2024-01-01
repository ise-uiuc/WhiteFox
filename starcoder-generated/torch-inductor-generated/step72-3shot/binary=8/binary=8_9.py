
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Conv -> bn -> relu
        # 2. FC -> bn -> relu
        layer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU())
        self.conv1 = layer
        self.conv2 = copy.deepcopy(layer)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        return v1 + v2
# Inputs to the model
x1 = torch.randn(1,3,224,224)
