
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 5, 2)
        self.conv2 = torch.nn.Conv2d(5, 5, 2)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 5, 2, 2)
