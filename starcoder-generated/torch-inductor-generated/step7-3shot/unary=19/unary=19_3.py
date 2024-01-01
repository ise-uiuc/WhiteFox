
class ModelNHWC(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(1, 12, (3, 3), stride=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(12)

        self.features = torch.nn.Sequential(
            # depthwise
            torch.nn.Conv2d(12, 12, (3, 3), stride=1, groups=12, bias=False),
            torch.nn.BatchNorm2d(12),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),

            # 5X5
            torch.nn.Conv2d(12, 10, (5, 5), stride=2, groups=1, bias=False)
        )

        self.fc = torch.nn.Linear(10, 8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.features(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x 

# Initializing the model
m = ModelNHWC()

# Inputs to the model
x = torch.randn(1, 1, 224, 224).to('cuda:0')

