
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=1),
            torch.nn.Conv2d(3, 64, kernel_size=5, padding=2)
        )
        self.layer2 = torch.nn.ReLU()
        self.layer3 = torch.nn.AdaptiveMaxPool2d((3, 3))
        self.layer4 = torch.nn.Flatten()
    def forward(self, x1):
        v1 = self.layer1(x1)
        v2 = self.layer2(v1)
        v3 = self.layer3(v2)
        v4 = self.layer4(v3)
        # x3 = torch.transpose(x2, 1, 2)
        v6 = torch.transpose(torch.transpose(v3, 2, 3), 1, 2)
        v7 = v6.flip(1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
