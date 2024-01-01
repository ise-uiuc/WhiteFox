
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3, stride=1, padding=3, dilation=2)
        self.layer1 = torch.nn.modules.pooling.MaxPool2d(3, stride=2, padding=1)
        self.layer2 = torch.nn.ReLU6(True)
    def forward(self, x1):
        v1 = self.conv(x1)
        for layer in self.children():
          v1 = layer(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
