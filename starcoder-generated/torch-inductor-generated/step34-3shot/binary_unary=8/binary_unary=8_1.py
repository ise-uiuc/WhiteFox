
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, (3, 5), stride=(2, 3), padding=(1, 2)),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
        )
    def forward(self, x1):
        x1 = self.features(x1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
