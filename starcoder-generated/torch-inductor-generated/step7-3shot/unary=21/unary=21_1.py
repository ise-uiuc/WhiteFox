
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=0, bias=True)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return v2

input = torch.randn(1, 16, 128, 128)
