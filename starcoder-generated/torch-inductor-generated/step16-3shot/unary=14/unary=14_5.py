
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0, bias=True)
        self.conv2 = torch.nn.Conv2d(16, 16, 4, stride=2, padding=1, bias=True)
        self.conv3 = torch.nn.Conv2d(16, 128, 2, stride=2, padding=1, bias=True)
        self.convT_1 = torch.nn.ConvTranspose2d(304, 16, 6, stride=2, padding=1, output_padding=1, dilation=2, bias=True)
        self.convT_2 = torch.nn.ConvTranspose2d(176, 64, 5, stride=2, padding=1, output_padding=1, dilation=2, bias=True)
        self.convT_3 = torch.nn.ConvTranspose2d(128, 4, 7, stride=3, padding=1, output_padding=1, dilation=2, bias=True)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(t1)
        t3 = self.conv3(t2)
        t5 = torch.cat([t2, t3], dim=1)
        x = self.convT_1(t5)
        x = self.convT_2(x)
        x = self.convT_3(x)
        return x
# Inputs to the model
x1 = torch.randn(2, 3, 256, 256)
