
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.Conv2d(32, 32, 3, 1, 1), torch.nn.Conv2d(32, 32, 3, 2, 3), torch.nn.Conv2d(32, 32, 3, 1, 1), torch.nn.Conv2d(32, 64, 3, 2, 0)])
    def forward(self, x27):
        x23, x24, x25, x26 = torch.split(x27, [1, 1, 1], dim=0)
        x37, x38 = torch.split(x33, [1, 1, 1], dim=0)
        x42, x43, x44 = torch.split(x41, [1, 1, 1], dim=0)
        x45, x46 = torch.split(x33, [1, 1, 1], dim=0)
        return x37, x38, x39, x40
# Inputs to the model
x1 = torch.randn(3, 1, 64, 64)
