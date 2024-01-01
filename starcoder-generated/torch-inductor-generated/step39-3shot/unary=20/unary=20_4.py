
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 3, kernel_size=(4, 5), stride=(6, 7), padding=(2, 2), output_padding=(1, 1), groups=1, bias=True, dilation=2, padding_mode=0)
        self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=1)
        self.conv3 = torch.nn.ConvTranspose2d(6, 2, kernel_size=(1, 2), stride=1, padding=0, bias=True, dilation=1, groups=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 18, 18)
