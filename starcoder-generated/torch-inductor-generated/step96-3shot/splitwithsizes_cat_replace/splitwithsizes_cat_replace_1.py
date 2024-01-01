
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.BatchNorm2d(32), torch.nn.ReLU(True), torch.nn.Conv2d(32, 8, 1, 1))
    def forward(self, v1):
        return torch.split(v1, [1, 1, 1], dim=1)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
