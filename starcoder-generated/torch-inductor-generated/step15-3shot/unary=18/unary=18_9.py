
class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=in_channels, bias=False, out_features=out_channels)
        self.linear2 = torch.nn.Linear(in_features=out_channels, bias=False, out_features=out_channels)
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.linear2(v2)
        v4 = torch.sigmoid(v3)
        return v4
in_channels = 3
out_channels = 128
kernel_size = 1
stride = 1
padding = 1
# Inputs to the model
x1 = torch.randn(3, 8, 12, 4)
