
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        t1 = v1 + v2
        v3 = self.conv2(t1)
        v4 = self.conv2(t1)
        t2 = v3 + v4
        v5 = torch.relu(t2)
        v6 = torch.relu(t2)
        v7 = v6 + v5
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
