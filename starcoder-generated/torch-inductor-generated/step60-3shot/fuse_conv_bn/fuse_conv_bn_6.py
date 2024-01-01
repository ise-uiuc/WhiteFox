
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 3, 3, bias=False)
        self.bn = torch.nn.BatchNorm2d(3, momentum=0.0, affine=False)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
# Inputs to the model
x = torch.randn(1, 2, 7, 7)
