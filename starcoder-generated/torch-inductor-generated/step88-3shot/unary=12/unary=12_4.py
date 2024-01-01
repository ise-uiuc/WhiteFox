
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.sigmoid(v1)
        v3 = F.relu(v2)
        v4 = torch.matmul(v3, v1)
        return v4.abs()
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
