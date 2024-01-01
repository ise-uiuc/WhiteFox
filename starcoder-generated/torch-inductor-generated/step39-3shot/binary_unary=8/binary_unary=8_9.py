
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 2, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.max_pool2d(v1, (3, 3), stride=2, padding=1)
        v3 = torch.max_pool2d(v1, 2, stride=1)
        v4 = torch.max_pool2d(v1, 3, stride=1, padding=0)
        v5 = torch.max_pool2d(v1, 2)
        v6 = torch.max_pool2d(v1)
        v7 = torch.max_pool2d(v1)
        v8 = v2 + v3 + v4 + v5 + v6 + v7
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
