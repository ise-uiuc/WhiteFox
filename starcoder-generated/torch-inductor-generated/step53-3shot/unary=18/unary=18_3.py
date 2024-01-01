
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_channels = 3
        self.output_channels = 128
        self.conv1 = torch.nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=self.output_channels, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 48, 48)
