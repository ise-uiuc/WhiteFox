
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=1)
    def forward(self, input_tensor):
        x1 = self.conv1(input_tensor)
        x2 = torch.sigmoid(x1)
        x3 = self.conv2(x2)
        x4 = torch.sigmoid(x3)
        x5 = self.conv3(x4)
        x6 = torch.sigmoid(x5)
        return x6
# Inputs to the model
input_tensor = torch.randn(1, 16, 64, 64)
