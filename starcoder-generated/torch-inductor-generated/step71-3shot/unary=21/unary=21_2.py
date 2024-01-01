
class ResBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, output_channels, kernel_size, padding=padding)
        self.conv2 = torch.nn.Conv2d(output_channels, output_channels, kernel_size, padding=padding)
        self.conv3 = torch.nn.Conv2d(output_channels, output_channels, kernel_size, padding=padding)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ResBlock(3, 32)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 256, 256)
