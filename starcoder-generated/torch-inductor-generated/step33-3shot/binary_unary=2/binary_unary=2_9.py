
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1)
    def forward(self, x1):
        v1 = self.conv(x1.permute(0, 1, 3, 2))
        v2 = v1 - -14
        v3 = F.relu(v2)
        v4 = v3.permute(0, 1, 3, 2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
