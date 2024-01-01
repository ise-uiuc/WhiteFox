
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = torch.nn.Conv2d(in_channels=17, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        y = self.conv2(x)
        return x
# Inputs to the model
x = torch.randn(2, 17, 8, 8)
