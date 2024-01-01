
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.t1 = torch.nn.Parameter(torch.ones(1, 8, 64, 64))
        self.t2 = torch.nn.Parameter(torch.ones(1, 8, 64, 64))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 * self.t1 
        v4 = v2 * self.t2
        v5 = v3 + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
