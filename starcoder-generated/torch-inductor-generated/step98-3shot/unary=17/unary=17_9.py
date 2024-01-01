
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 32, (3, 3), (2, 3), (1, 1), (1, 1))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = v2.transpose(1, 2)
        v4 = F.avg_pool2d(v3, 3, stride=2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 16, 24)
