
class BasicBlock(torch.nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = torch.nn.functional.gelu(x_1)
        x_2 = self.conv2(x_1)
        x_3 = torch.tanh(x_2)
        x_3 = self.sigmoid(x_3)
        x_final = x * x_3
        return x_final
class TanhActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.tanh(x)
        return x
class Model(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, 7, stride=2, padding=3)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.activ = TanhActivation()
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.block1 = BasicBlock(kernel_size=3, in_channels=64, out_channels=128)
        self.block2 = BasicBlock(kernel_size=3, in_channels=128, out_channels=256)
        self.block3 = BasicBlock(kernel_size=3, in_channels=256, out_channels=512)
        self.avgpoo = torch.nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.gelu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.activ(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpoo(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
in_channels = 1
num_classes = 1000
# Inputs to the model
x = torch.randn(128, in_channels, 224, 224)
