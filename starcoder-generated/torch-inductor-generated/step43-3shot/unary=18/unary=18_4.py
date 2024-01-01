
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7))
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 15), stride=(1, 1), padding=(0, 13))
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = torch.sigmoid(x2)
        x4 = self.conv2(x3)
        x5 = torch.sigmoid(x4)
        x6 = torch.cat([x1, x5], dim=1)
        return x6
# Inputs to the model
x1 = torch.randn(1, 1, 256, 288)
