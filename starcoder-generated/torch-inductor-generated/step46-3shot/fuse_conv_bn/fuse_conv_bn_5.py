
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 512, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn = torch.nn.BatchNorm2d(512)
        self.relu = torch.nn.ReLU6()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn(x))
        return x
# Inputs to the model
x = torch.randn(1, 3, 256, 256)
