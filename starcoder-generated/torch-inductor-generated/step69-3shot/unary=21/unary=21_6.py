
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 32, 3, stride=3, padding=0, bias=True)
        self.gelu = nn.GELU()
    def forward(self, x):
        return self.gelu(self.conv(x))
# Inputs to the model
x = torch.randn(1, 16, 100, 100)
