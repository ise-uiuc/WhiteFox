
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(16, 32, (2, 2), stride=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        v4 = torch.cat([v3, v3], dim=1)
        return torch.cat([v4, v4], dim=1)
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
