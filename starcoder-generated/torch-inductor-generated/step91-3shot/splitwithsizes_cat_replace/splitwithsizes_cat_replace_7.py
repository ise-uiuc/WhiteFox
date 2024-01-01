
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 3, 2, 1, bias=False)
    def forward(self, x):
        x = torch.split(x, [2,2,2,2], dim=1)[0]
        x = torch.cat([x, x, x, x], dim=1) # pattern doesn't exist because dim=2 here
        return self.conv(x)
# Inputs to the model
x = torch.randn(128, 32, 8, 8)
