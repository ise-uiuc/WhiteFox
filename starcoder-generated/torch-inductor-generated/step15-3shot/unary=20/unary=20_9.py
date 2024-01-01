
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(input_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=0, bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(3, 10, 34, 94)
