
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 5, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v3 = F.sigmoid(v3 + 2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 3, 100, 100)
