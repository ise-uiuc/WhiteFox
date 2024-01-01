
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=True)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=True)
    def forward(self, x1, x2):
        v1 = torch.add(x1, self.conv1(x2))
        v2 = torch.relu(v1)
        v3 = v2 + self.conv3(x1)
        v4 = torch.relu(v3)
        v5 = v4 + self.conv2(x2)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
