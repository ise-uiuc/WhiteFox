
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 2, stride=1, padding=4)
        self.conv2 = torch.nn.Conv2d(8, 32, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = [v1] * 5
        v3 = torch.cat(v2, 1)
        v4 = self.conv2(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 24, 24)
