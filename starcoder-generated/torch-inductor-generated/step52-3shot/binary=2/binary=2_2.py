
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 512, kernel_size=(9, 10))
        self.conv2 = torch.nn.Conv2d(512, 512, kernel_size=(10, 10))
    def forward(self, x3):
        v1 = self.conv1(x3)
        v2 = self.conv2(v1)
        v3 = v2 - 1.0
        return v3

# Inputs to the model
x3 = torch.randn(1, 1, 56, 100)
