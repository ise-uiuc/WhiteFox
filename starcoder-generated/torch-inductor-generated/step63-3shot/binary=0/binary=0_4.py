
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 1, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
    def forward(self, x1, other1=1, other2=1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        t1 = torch.randn(v2.shape)
        v3 = v2 + other1 + other2
        v4 = v3 + t1
        return v4
# Inputs to the model
x1 = torch.randn(4, 2, 224, 224)
