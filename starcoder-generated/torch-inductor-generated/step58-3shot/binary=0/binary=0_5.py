
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=0)
    def forward(self, x1, other=1, other1=2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 + v2
        if other == False:
            other = torch.randn(v3.shape)
        v4 = v3 + other
        v5 = v4 + other1
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
