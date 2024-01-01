
class ConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, kernel_size=(4,4,3), stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.conv2 = torch.nn.Conv2d(2, 6, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)

    def forward(self, x):
        # pass through relu
        x = self.conv1(x)
        x = torch.relu(x)
        # pass through relu
        x = self.conv2(x)
        x = torch.relu(x)
        return x.flatten()
# Inputs to the model
x = torch.randn(128, 3, 64, 624)
