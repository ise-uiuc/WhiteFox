
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 16, (1, 1), (1, 1), (0, 0), 1, 1, False, False, 2, False)
        self.conv = torch.nn.Conv2d(16, 1, (1, 1))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = self.conv(v2)
        return torch.sigmoid(v3)
