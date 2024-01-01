
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
    def forward(self, input_tensor):
        x1 = self.conv1(input_tensor)
        x2 = torch.sigmoid(x1)
        x3 = self.conv2(x2)
        x4 = torch.sigmoid(x3)
        return x4
# Inputs to the model
input_tensor = torch.randn(1, 16, 16, 16)
