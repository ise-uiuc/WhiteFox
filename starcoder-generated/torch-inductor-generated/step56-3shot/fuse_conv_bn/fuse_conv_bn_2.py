
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__( )
        self.features = nn.Sequential(
            nn.Conv2d(2, 1, 3),
            nn.BatchNorm2d(1),
            nn.MaxPool2d(2),  # stride equal to kernel_size
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        x = self.features(x)
        return x
# Inputs to the model
x = torch.randn(2, 2, 5, 6)
