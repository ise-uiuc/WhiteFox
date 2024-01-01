
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
    def forward(self, x):
        return torch.cat([self.conv1(x), self.conv2(x)], 0)
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
