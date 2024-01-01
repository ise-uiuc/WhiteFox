
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 4, groups=32, bias=True)
        self.conv1 = torch.nn.Conv2d(32, 32, 3, bias=False)
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.conv1(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 16, 16)
