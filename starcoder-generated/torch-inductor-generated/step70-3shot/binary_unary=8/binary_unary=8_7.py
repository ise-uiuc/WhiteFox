
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * v1 * v1
        v3 = torch.max(v2, dim=1, keepdim=False)[0]
        v4 = torch.tanh(v2)
        v5 = self.conv1(x1)
        v6 = self.conv1(x1)
        v7 = self.conv1(x1)
        v8 = self.conv1(x1)
        v9 = v5 + v6 + v7 + v8
        v10 = torch.relu(v5 - 0.0158, 0)
        return v9, v10
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
