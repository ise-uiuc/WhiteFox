
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, 256)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, 256)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        return v1, v2
# Inputs to the model
x1 = torch.randn(1, 3, 352, 352)
x2 = torch.randn(1, 3, 352, 352)
