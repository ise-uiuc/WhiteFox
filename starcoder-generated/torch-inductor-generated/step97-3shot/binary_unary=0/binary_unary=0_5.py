
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.batch1 = torch.nn.BatchNorm2d(64)
    def forward(self, inputs):
        x = self.conv(inputs)
        x_bn = self.batch1(x)
        x = torch.relu(x_bn)
        return x
# Inputs to the model
inputs = torch.randn(1, 64, 224, 224)
