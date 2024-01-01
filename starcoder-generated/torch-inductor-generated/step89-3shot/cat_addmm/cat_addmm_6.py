
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 5, 2, bias=True, groups=1)
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return x
# Inputs to the model
x = torch.randn(4, 3, 100, 100)
