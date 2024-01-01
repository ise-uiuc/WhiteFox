
class Model(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(num_features, 256, 3, stride=1, padding=1)
        self.act1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(256, 8, 1, stride=1, padding=0)
        self.add1 = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.act1(v1)
        v3 = self.conv2(v2)
        v4 = self.add1(v3)
        return v4
# Inputs to the model
num_features = 16
x1 = torch.randn(1, num_features, 48, 48)
