
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 3, kernel_size=3),
            nn.BatchNorm2d(3),
            nn.MaxPool2d(2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 4, kernel_size=2)
        )

    def forward(self, x):
        return self.model(x)
# Inputs to the model
x = torch.randn(2, 2, 5, 6)
