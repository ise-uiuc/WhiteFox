
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = torch.ones(1, x.shape[1], x.shape[2], x.shape[3])
        v2 = self.conv1(x)
        v3 = self.conv2(x)
        v4 = v1 + v2
        v5 = torch.ones(1, x.shape[1], x.shape[2], x.shape[3])
        v6 = v3 + v5
        v7 = v4 + v6
        return v7
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
