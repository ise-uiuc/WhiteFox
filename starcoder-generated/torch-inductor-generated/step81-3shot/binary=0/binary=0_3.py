
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=1, padding1=1, padding2=1):
        v1 = self.conv1(x1)
        v2 = self.conv1(v1)
        if padding1 == None:
            padding1 = torch.randn(v2.shape)
        if padding2 == None:
            padding2 = torch.randn(v2.shape)
        v3 = v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
