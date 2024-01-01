
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3), torch.nn.ReLU(inplace = True))
    def forward(self, x1):
        v1 = self.block(x1)
        v, v1 = torch.split(v1, [1, 1, 1], dim=1)
        x2 = self.block(v)
        return (x1 * x2, (v1 * x1, v1 * x2), v1)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
