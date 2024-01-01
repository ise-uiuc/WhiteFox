
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        model = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 20, 5, 1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)
# Inputs to the model
x = torch.randn(2, 1, 28, 28)
