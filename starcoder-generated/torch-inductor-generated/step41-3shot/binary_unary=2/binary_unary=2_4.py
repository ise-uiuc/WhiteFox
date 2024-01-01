
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, 1, stride=1, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(2, 4, 1, stride=3, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        t1 = v1 - 0.6
        v2 = self.conv2(v1)
        t2 = v2 + x1
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 100, 100)
