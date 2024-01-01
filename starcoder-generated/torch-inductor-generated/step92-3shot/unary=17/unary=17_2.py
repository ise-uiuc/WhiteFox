
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(6, 64, 3, stride=1, padding=1)
        self.conv1 = nn.Conv2D(64, 128, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(128, 1024, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2D(1024, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = nn.ReLU()(v1)
        v3 = self.conv1(v2)
        v4 = nn.ReLU()(v3)
        v5 = self.conv2(v4)
        v6 = nn.ReLU()(v5)
        v7 = self.conv3(v6)
        return nn.Sigmoid()(v7)
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
#Model end
