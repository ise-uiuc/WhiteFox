
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, bias=False)
    def forward(self, x1):
        x1 = self.conv(x1)
        y = x1[:,:,1:-1,1:-1].sum(1)
        return torch.flatten(y, 1)
# Inputs to the model
x1 = torch.randn(1, 3, 6, 6)
