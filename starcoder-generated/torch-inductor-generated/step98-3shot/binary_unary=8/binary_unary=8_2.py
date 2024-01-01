
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Linear(1792, 4096)
    def forward(self, x1):
        v1 = self.avgpool(x1)
        v2 = torch.flatten(v1, 1)
        v3 = self.fc1(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
