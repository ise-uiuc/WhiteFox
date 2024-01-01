
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 64, 1, stride=1, padding=0)
    def forward(self, x1, dim=1, dim1=2):
        v1 = self.conv(x1)
        v2 = torch.squeeze(v1, dim)
        v3 = torch.squeeze(v2, dim1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 8, 8)
