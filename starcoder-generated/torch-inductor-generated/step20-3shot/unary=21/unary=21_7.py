
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = 1, out_channels = 25, kernel_size = 7, stride = 2, padding = 2, bias = False)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = F.leaky_relu(v1)
        return v2
# Inputs to the model
input = torch.randn(1, 1, 224, 224)
