
class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = torch.nn.Conv2d(3, 6, 3, padding=1, bias=False)
        self.conv_transpose2d1 = torch.nn.ConvTranspose2d(6, 3, 3, padding=1, output_padding=0, bias=False)
        self.bn = torch.nn.BatchNorm2d(3, eps=0.00001000000050000000020, momentum=0.0, affine=True, track_running_stats=True)
        self.relu6_module = torch.nn.Sequential(
            torch.nn.Conv2d(3, 10, kernel_size=(1, 1), stride=(1, 1), bias=False),
            torch.nn.BatchNorm2d(10, eps=0.00001000000050000000020, momentum=0.0, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=False)
        )

    def forward(self, x1):
        v1 = self.conv2d1(x1)
        v2 = self.bn(v1)
        v3 = torch.nn.functional.relu6(v2)
        v4 = self.conv_transpose2d1(v3)
        v5 = self.relu6_module(v4)
        return v5

# Input to the model
x1 = torch.randn(1, 3, 64, 64)
