
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 3)
        self.pooling = torch.nn.MaxPool1d(1)

    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.pooling(v1)
        v3 = F.adaptive_avg_pool2d(v2, (1, 2))
        return v3
x1 = torch.randn(1, 1, 3, 3)
