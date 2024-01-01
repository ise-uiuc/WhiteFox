
class model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding): 
        super(model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x): 
        v1 = self.conv(x)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
in_channels, out_channels, kernel_size, stride, padding = 3,128,1,1,1
