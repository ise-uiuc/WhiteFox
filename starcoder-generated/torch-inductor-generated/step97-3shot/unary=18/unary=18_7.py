
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=100, out_channels=40, kernel_size=5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, stride=2, padding=0)
    def forward(self, x2):
        v1 = self.conv1(x2)
        v2 = torch.sigmoid(v1)
        v5 = self.conv2(v2)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x2 = torch.randn(1, 100, 100, 100)
