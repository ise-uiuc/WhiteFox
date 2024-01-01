
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 32, 3, stride=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.relu(v1)
        v3 = v2.transpose(3, 2)
        v4 = F.avg_pool2d(v3, 2, stride=2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
