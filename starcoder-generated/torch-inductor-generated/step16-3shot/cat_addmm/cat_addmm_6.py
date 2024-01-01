
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
    def forward(self, x):
        x = self.conv(x)
        x = torch.cat((x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2, 2, 2)
