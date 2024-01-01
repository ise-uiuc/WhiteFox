
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 12, 1),
            torch.nn.BatchNorm2d(12)
        )
    def forward(self, x):
        return self.block(x)
# Inputs to the model
x = torch.randn(1, 4, 4, 4)
