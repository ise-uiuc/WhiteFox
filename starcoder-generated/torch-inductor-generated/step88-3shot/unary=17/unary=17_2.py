
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, (3, 3), 1, (1, 1), (1, 0))
        self.conv_1 = torch.nn.Conv2d(16, 32, (3, 3), 1, (1, 1), (1, 0))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.relu(v1)
        v3 = self.conv_1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
