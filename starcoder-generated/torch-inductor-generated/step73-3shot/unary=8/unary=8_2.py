
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(128, 12, 4, stride=3, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v15 = torch.unsqueeze(torch.unsqueeze(v1, dim=3), dim=3)
        v2 = torch.matmul(v1, v15) + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
x1 = torch.randn(1, 128, 13, 13)
