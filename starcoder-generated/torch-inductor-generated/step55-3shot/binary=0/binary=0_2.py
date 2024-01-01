
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 8, 3, stride=1, padding=1)
    def forward(self, x1, other1=None, other2=None):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        if other1 == None:
            other1 = torch.randn(v3.shape)
        if other2 == None:
            other2 = torch.randn(v2.shape)
        v4 = v3 + other1
        v5 = v2 + other2
        return v4
# Inputs to the model
x1 = torch.randn(3, 3, 64, 64)
