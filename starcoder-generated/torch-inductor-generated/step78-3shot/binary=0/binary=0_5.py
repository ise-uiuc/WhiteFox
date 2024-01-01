
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(1, 22, 3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(1, 20, 7, stride=1, padding=0)
    def forward(self, x1):
        v0 = self.conv0(x1)
        v1 = self.conv1(v0)
        v2 = v1 + v0
        v3 = v2.relu()
        v4 = v2.relu().tanh()
        v5 = v2 + v2.tanh()
        v6 = v3 + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 8, 8)
