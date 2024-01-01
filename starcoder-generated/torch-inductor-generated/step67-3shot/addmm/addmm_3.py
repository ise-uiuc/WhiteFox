
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 3)
        r = torch.mm(torch.eye(3), torch.eye(3))
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
        self.conv3 = torch.nn.Conv2d(3, 1, 3)
    def forward(self, x1, x2=None):
        if x2 is not None:
            r = x1
        else:
            r = x2
        v1 = self.conv1(r)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        return v3 + v2
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
x2 = torch.randn(1, 3, 3, 3)
