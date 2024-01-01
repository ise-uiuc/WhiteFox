
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=5, kernel_size=[3, 3], stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=5, out_channels=8, kernel_size=[1, 1], stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=8, out_channels=7, kernel_size=[1, 1], stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v2 = torch.mul(v2, self.conv3(v2))
        v3 = torch.sigmoid(v2)
        return v3

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
