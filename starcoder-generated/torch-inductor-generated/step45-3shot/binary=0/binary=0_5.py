
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 6, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(6, 6, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(6, 6, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(6, 6, 1, stride=1, padding=1)
    def forward(self, x1, other=None):
        v1 = self.conv2(self.conv1(x1))
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(9, 32, 64, 64)
