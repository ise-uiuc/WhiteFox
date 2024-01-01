
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(1, 1), stride=(1, 1), padding=(0,), dilation=(1,), groups=1, bias=True, padding_mode='zeros')
        self.flatten = torch.nn.Flatten(1, -1)
        self.softmax = torch.nn.Softmax(dim=-1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = self.conv(v1)
        v3 = self.flatten(v2)
        v4 = self.softmax(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
