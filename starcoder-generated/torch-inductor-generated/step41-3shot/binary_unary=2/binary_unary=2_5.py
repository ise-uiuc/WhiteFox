
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        t1 = v1 - 1
        v2 = self.conv2(t1)
        v2 = v2 - 2
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)
