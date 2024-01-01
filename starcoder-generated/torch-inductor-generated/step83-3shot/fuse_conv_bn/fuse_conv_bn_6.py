 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 2, groups=3)
        self.bn1 = torch.nn.BatchNorm3d(3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return x 
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
