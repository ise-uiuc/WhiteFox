
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
                                          torch.nn.BatchNorm2d(num_features=16),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
                                          torch.nn.BatchNorm2d(num_features=32),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                                          torch.nn.BatchNorm2d(num_features=64),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1))
    def forward(self, x1):
        v1 = self.layer1.forward(x1)
        v2 = self.layer2.forward(v1)
        v3 = self.layer3.forward(v2)
        v4 = self.layer4.forward(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
