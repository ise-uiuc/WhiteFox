
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, 11, bias=False)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        v3 = self.avgpool(v2)
        v4 = self.flatten(v3)
        v5 = self.fc(v4)
        return v5
# Inputs to the model
input = torch.randn(1, 3, 224, 224)
