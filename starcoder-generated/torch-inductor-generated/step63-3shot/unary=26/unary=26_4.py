
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1024, 384, 3, stride=1, dilation=1, groups=1, padding=1, output_padding=0)
        self.conv = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 384, (1,1), stride=(1,1), bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 384, (1,1), stride=(1,1), bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, (3,3), stride=(2,2), padding=(1,1), bias=False)
        )
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * -0.01
        x4 = torch.where(x2, x1, x3)
        x5 = self.conv(x4)
        return torch.nn.functional.adaptive_avg_pool2d(x5, (1, 1))
# Inputs to the model
x = torch.randn(32, 1024, 7, 7)
