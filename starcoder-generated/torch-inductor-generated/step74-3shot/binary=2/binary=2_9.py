
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(24, 48, (1, 5), stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(48, 24, (3, 1), stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - 10
        return v3
# Inputs to the model
x1 = torch.randn(1, 24, 100, 50)
