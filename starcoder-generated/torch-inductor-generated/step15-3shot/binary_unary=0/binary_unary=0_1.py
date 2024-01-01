
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.transpose(v1, 1, 2)
        v3 = torch.relu(v1)
        v4 = torch.matmul(v3, v2)
        v5 = v4 + v3
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
