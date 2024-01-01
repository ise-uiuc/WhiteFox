
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = torch.nn.Conv2d(3, 3, 1)
        self.conv1b = torch.nn.Conv2d(3, 3, 2)
    def forward(self, x1):
        v1 = v2 = x1
        v3a = self.conv1a(x1)
        v4a = F.relu(v3a)
        v5a = torch.squeeze(v4a, 2)
        v3b = self.conv1b(x1)
        v4b = F.relu(v3b)
        v5b = torch.squeeze(v4b, 2)
        v6 = v5a - v5b
        return v6
# Inputs to the model
x1 = torch.randn(3, 3, 3, 3)
