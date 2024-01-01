
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 7, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(7, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
