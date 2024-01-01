
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.Conv2d(32, 32, 5, 2, 2), torch.nn.Conv2d(32, 3, 3, 1, 1))
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, 1, 1), torch.nn.Conv2d(3, 3, 3, 1, 1), torch.nn.Conv2d(3, 3, 3, 1, 1))
    def forward(self, v1):
        return self.conv(v1)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
