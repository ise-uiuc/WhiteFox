
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, 2),
            torch.nn.UpsamplingBilinear2d(scale_factor=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.Conv2d(1, 1, 1),
            torch.nn.ReLU(True),
        )
    def forward(self, x):
        x = self.block1(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
