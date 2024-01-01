
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=30, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=2, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
