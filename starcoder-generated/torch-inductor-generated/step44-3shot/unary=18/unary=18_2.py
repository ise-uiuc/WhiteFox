
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(100, 300, 1)
        self.conv2 = torch.nn.Conv2d(300, 300, 1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 100, 100)
