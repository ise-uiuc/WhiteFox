
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1, 1, 0)
    def forward(self, x1):
        v1 = torch.sigmoid(self.conv1(x1))
        v2 = torch.sigmoid(self.conv2(v1))
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
