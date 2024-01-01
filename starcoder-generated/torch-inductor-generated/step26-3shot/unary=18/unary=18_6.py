
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, 3, 1, 1),
            torch.nn.ReLU()
        )
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
