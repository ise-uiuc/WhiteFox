
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.tanh(v1 + x1)
        v3 = torch.relu(v1 + x1)
        v4 = torch.relu(v1 + x1)
        v5 = torch.sigmoid(v1 + x1)
        v6 = v2 + v3 + v4 + v5
        v7 = torch.relu(v1 + x1)
        v8 = torch.tanh(v1 + x1)
        v9 = torch.relu(v1 + x1)
        v10 = v7 + v8 + v9
        res = torch.relu(v10)
        return res
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
