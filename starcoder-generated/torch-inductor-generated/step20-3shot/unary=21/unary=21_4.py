
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(29, 27, kernel_size=3, stride=1, padding=0, bias=True)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
input = torch.randn(1, 29, 33, 33)
