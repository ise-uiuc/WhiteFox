
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad1 = torch.nn.ReflectionPad2d(1)
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=8, bias=True, padding=2, kernel_size=5, stride=1
        )
        self.pad2_1 = torch.nn.ReflectionPad2d(1)
        self.conv2_1 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, bias=True, padding=2, kernel_size=5, stride=1
        )
        self.pad2_2 = torch.nn.ReflectionPad2d(1)
        self.conv2_2 = torch.nn.Conv2d(
            in_channels=16, out_channels=16, bias=True, padding=2, kernel_size=5, stride=1
        )
        self.pad3_1 = torch.nn.ReflectionPad2d(1)
        self.conv3_1 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, bias=True, padding=2, kernel_size=5, stride=1
        )
        self.pad3_2 = torch.nn.ReflectionPad2d(1)
        self.conv3_2 = torch.nn.Conv2d(
            in_channels=32, out_channels=32, bias=True, padding=2, kernel_size=5, stride=1
        )
    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.pad2_1(x)
        x = self.conv2_1(x)
        x = torch.tanh(x)
        x = self.pad2_2(x)
        x = self.conv2_2(x)
        x = torch.tanh(x)
        x = self.pad3_1(x)
        x = self.conv3_1(x)
        x = torch.tanh(x)
        x = self.pad3_2(x)
        x = self.conv3_2(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 40, 40)
