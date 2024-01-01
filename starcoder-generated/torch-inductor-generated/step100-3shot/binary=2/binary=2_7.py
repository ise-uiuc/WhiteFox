
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, kernel_size=32, stride=31, padding=17)
        self.conv2 = torch.nn.Conv2d(1, 16, kernel_size=(13, 4), stride=(27, 3), padding=(9, 18))
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1 - 12.1
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 3, 16, 32)
