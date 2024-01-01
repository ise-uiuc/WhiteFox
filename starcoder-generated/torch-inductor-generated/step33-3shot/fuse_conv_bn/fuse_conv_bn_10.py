
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512, 0.8),
            torch.nn.Conv2d(512, 512, 1)
        )
        self.relu = torch.nn.ReLU()
    def forward(self, x3):
        y0 = self.block0(x3)
        y = self.relu(y0)
        return y
# Inputs to the model
x3 = torch.randn(1, 512, 30, 30)
