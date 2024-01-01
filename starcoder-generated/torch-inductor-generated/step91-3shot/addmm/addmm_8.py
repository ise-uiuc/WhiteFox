
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 32, kernel_size=1)
        self.conv1 = torch.nn.Conv2d(32, 32, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(32, 3, kernel_size=1)
    def forward(self, x):
        v1 = self.conv0(x)
        v2 = self.conv1(v1)
        return self.conv2(v2)
# Inputs to the model
x = torch.randn(1, 3, 10, 10)
