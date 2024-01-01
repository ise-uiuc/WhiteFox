
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=11, out_channels=48, kernel_size=(2, 3), stride=(3, 2), padding=(1, 0))
        self.conv2 = torch.nn.Conv2d(in_channels=48, out_channels=512, kernel_size=1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 11, 32, 64)
