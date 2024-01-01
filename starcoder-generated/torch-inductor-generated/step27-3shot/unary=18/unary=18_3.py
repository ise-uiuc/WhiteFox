
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=1)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=5, stride=2)
        self.conv3 = torch.nn.Conv2d(4, 8, kernel_size=(1, 2), stride=(2, 1), padding=(0, 1), dilation=(1, 2))
    def forward(self, x1):
        x = torch.randn(1, 4, 5, 5)
        v1 = self.conv1(x)
        v1 = self.conv2(v1)
        v2 = self.conv3(x)
        return v1, v2
# Inputs to the model
x1 = torch.randn(1, 4, 5, 5)
