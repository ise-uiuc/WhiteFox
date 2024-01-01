
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.sigmoid()
        t1 = v1.flatten(start_dim=1)
        v3 = torch.matmul(t1, t1.T)
        v4 = torch.matmul(v3, t1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
