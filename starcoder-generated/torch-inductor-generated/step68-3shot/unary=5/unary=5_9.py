
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(6, 6, 3, stride=3, padding=0)
    def forward(self, x1):
        v1 = v4 = self.conv_transpose(x1)
        v2 = torch.cat([v4, v4, v4], 1)
        v3 = torch.cat([v2, v2, v2], 1)
        v5 = torch.cat([v3, v3, v3], 1)
        v6 = torch.cat([v5, v5, v5], 1)
        return v6
# Inputs to the model
x1 = torch.randn(1, 6, 12, 12)
