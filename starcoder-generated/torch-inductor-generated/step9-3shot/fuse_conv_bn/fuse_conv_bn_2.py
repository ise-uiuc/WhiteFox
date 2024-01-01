
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 3, 3)
    def forward(self, x0):
        x0 = self.conv(x0)
        return torch.cat(4 * [x0], 1)
# Inputs to the model
x0 = torch.randn(1, 6, 4, 4)
