
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, 2, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(2, 3, 2, 1)
    def forward(self, x1):
        x = self.conv1(x1)
        return self.conv2(x).reshape(1, 9)
# Inputs to the model
x1 = torch.randn(1, 3, 3, 5)
