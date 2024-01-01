
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3)
        self.conv2 = torch.nn.Conv2d(1, 1, 3)
    def forward(self, x3):
        v1 = self.conv1(x3)
        v2 = self.conv2(x3)
        v2 = v2 - 1.0
        return v2
# Inputs to the model
x3 = torch.randn(1, 1, 7, 7)
