
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.fc = torch.nn.Linear(in_features=12288, out_features=10)
    def forward(self, x):
        v0 = torch.flatten(x, 1)
        v1 = self.conv(x)
        v2 = v1.mean([2, 3])
        v3 = torch.cat((v0, v2), 1)
        v4 = self.fc(v3)
        v5 = torch.softmax(v4, dim=1)
        return v5
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
