
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 7, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(7, 15, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 + torch.randn(v1.size())
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v2 + torch.randn(v4.size())
        v6 = v4 + torch.randn(v4.size())
        v7 = torch.relu(v5)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
