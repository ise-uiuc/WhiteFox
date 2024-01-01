
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(1,1,stride = 2, padding = 10, bias = False)
        self.bn = nn.BatchNorm2d(1, momentum=0.0001, affine=True)
        self.relu = nn.LeakyReLU(inplace=False)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.bn(x2)
        x4 = self.relu(x3)
        return x4
# Input tensor for this model
x1 = torch.randn(1, 1, 50, 200)
