
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = list([v2, x1])
        v4 = torch.cat(v3, 1)
        v5 = self.conv2(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 256, 256)
