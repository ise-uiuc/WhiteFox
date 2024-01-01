
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
    def forward(self, x1, other=None):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        if other!= None:
            v2 += v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
t = torch.randn(32, 29, 29)
