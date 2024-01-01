
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(23, 8, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 128, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = v2 - 1.7E-2
        return v3
# Inputs to the model
x = torch.randn(1, 23, 23, 23)
