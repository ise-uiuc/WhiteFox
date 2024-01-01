
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 3, 2, stride=2, padding=2, groups=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        return torch.squeeze(v3, dim=0)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
