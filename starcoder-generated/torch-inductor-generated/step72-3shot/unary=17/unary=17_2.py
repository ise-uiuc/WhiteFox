
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=(5, 12), mode='nearest'),
            torch.nn.Conv2d(3, 9, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU(),
            torch.nn.Conv2d(9, 12, 4, stride=1, padding=1),
            torch.nn.MaxPool2d(4, 4, 4),
        )
    def forward(self, x1):
        return self.features(x1)
# Inputs to the model
x1 = torch.randn(1, 3, 8, 8)
