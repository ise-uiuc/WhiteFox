
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=3)
    def forward(self, x1, other=True, padding1=None):
        v1 = self.relu(self.conv(x1))
        if padding1 == None:
            x2 = self.conv1(v1)
        v2 = v1 + x2
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
