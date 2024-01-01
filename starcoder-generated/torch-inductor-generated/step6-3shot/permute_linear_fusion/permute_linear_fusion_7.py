
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1, 3, (3, 3))
        self.flatten = torch.nn.Flatten()
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.flatten(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
