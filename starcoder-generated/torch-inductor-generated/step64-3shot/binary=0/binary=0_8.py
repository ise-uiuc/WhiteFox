
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(12, 18, 1, stride=2)
        self.conv3 = torch.nn.Conv2d(36, 2, 1)
        self.conv4 = torch.nn.Conv2d(12, 6, 1)
    def forward(self, x1, padding1=None, padding2=None, padding3=None):
        v1 = self.conv1(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        if padding2 == None:
            padding2 = torch.randn(v1.shape)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        if padding3 == None:
            padding3 = torch.randn(v1.shape)
        v4 = self.conv4(v2)
        return (v4, v3)
# Inputs to the model
x1 = torch.randn(4, 3, 64, 64)
