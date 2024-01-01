
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.randn(8)
        v3 = v1 * v2
        v4 = torch.cat([v3, v3], dim=1)
        v5 = v4[:, 0:size]
        v6 = torch.cat([v4, v5], dim=1)
        return v6

# Inputs to the model
x1 = torch.randn(8, 3, 64, 64)
