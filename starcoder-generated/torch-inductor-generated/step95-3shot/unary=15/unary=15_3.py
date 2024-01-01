
class Residual(nn.Module):
    def __init__(self, in_features):
        super(Residual, self).__init__()
        conv_block = [nn.Conv2d(in_features, in_features, kernel_size=3, padding=1), nn.BatchNorm2d(in_features), nn.ReLU(inplace=True), nn.Conv2d(in_features, in_features, kernel_size=3, padding=1), nn.BatchNorm2d(in_features)]
        self.conv_block = nn.Sequential(*conv_block)
    def forward(self, x):
        return x + x + self.conv_block(x)    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, padding=1)
        self.res1 = Residual(128)
        self.res2 = Residual(128)
        self.res3 = Residual(128)
        self.res4 = Residual(128)
    def forward(self, x1):
        v1 = F.relu(self.conv1(x1))
        v2 = self.res1(v1) + v1
        v3 = self.res2(v2) + v2
        v4 = self.res3(v3) + v3
        v5 = self.res4(v4) + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 512, 512)
