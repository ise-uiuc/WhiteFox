
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 76, (1, 9), stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(76, 1, (3, 1), stride=1, padding=1)
    def forward(self, x0):
        v0 = self.conv1(x0)
        v1 = self.conv2(v0)
        v2 = v1 - 3.7
        return v2
# Inputs to the model
x0 = torch.randn(1, 1, 26, 27)
