
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, groups=2, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.nn.functional.sigmoid(self.conv1(x1))
        v2 = torch.nn.functional.relu(v1)
        v3 = torch.tanh(self.conv2(v2))
        v4 = torch.nn.functional.conv_transpose2d(v3, weight=self.conv3.weight)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
