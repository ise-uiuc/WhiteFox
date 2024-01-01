
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 1)
        self.conv2 = torch.nn.Conv2d(16, 16, 1)
    def forward(self, x):
        x = self.conv1(self.conv2(x))
        return x
# Inputs to the model
x = torch.randn(1, 16, 8, 8)
