
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 1, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(128, 3, 2, stride=1, padding=1)
    def forward(self, x1, other=1, padding1=None, bias1=None):
        v1 = self.conv1(x1)
        if other == 1 and padding1 == None and bias1 == None:
            other = torch.randn(v1.shape)
        v2 = self.conv2(v1)
        if padding1 == None and bias1 == None:
            padding1 = torch.randn(v2.shape)
        v3 = v2 + other
        if bias1 == None:
            bias1 = torch.randn(v3.shape)
        v4 = v3 + padding1
        return v4
# Inputs to the model
x1 = torch.randn(1, 128, 256, 256)
