
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(11, 48, 2, 3, 1)
        self.conv2 = torch.nn.Conv2d(48, 512, 1, 1, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.sigmoid(v2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 11, 32, 64)
