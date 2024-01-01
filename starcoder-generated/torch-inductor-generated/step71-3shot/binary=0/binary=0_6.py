
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 20, 11, stride=4, padding=0)
        self.conv1 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x1, other=4.4, t1=torch.randn(1,1,1,1)):
        v1 = self.conv(x1)
        if t1.shape == v1.shape:
            v2 = t1 + v1
        else:
            v2 = v1 + t1
        v3 = v2 + 4.4
        v4 = v3 * 2
        return v4
# Inputs to the model
x1 = torch.randn(1, 10, 12, 12)
