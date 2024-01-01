
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = torch.nn.MaxPool2d(kernel_size = 3, stride=2, ceil_mode = True )
        self.conv= torch.nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=3, stride=1, groups = 4, padding=1)
    def forward(self,x):
        v1 = self.conv(x)
        v2 = self.max_pool(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 3, 256, 256)
