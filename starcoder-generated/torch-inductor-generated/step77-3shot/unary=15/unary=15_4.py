
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = torch.nn.Conv2d(249, 124, 1, stride=1, padding=0)
        self.conv2d_2a = torch.nn.Conv2d(124, 20, 1, stride=1, padding=0)
        self.conv2d_2b = torch.nn.Conv2d(124, 20, 1, stride=1, padding=0)
    def forward(self, x1):
        v1a = self.conv2d_1(x1)
        v1b = self.conv2d_1(x1)
        v2a = v1a.max(dim=1, keepdim=True)[0]
        v2b = v1b.max(dim=1, keepdim=True)[0]
        v3a = v2a.repeat(1, 20, 1, 1)
        v3b = v2b.repeat(1, 20, 1, 1)
        v4a = torch.relu(v3a)
        v4b = torch.relu(v3b)
        v5a = self.conv2d_2a(v4a)
        v5b = self.conv2d_2b(v4b)
        return (v5a, v5b)
# Inputs to the model
x1 = torch.randn(1, 249, 32, 32)
