
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.Conv2d(5, 10, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )
        self.pool = nn.MaxPool2d(2, stride=3)
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x
# Inputs to the model
x = torch.randn(2, 1, 10, 11)
