
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, stride=3)
        self.conv2 = torch.nn.Conv2d(3, 4, 3, stride=3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        return (v1, v2)
# Inputs to the model
x = torch.randn(1, 3, 25, 25)
