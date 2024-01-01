
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=1, padding1=None, padding2=None):
        v1 = self.conv1(x1)+other
        if padding2 == None:
            padding2 = torch.randn(x1.shape).float()
        v2 = self.conv2(v1)+other
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
