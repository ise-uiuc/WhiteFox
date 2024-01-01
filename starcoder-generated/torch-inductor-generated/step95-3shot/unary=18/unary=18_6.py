
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(9, 5, 5, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv1(v1)
        v4 = torch.sigmoid(v3)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
