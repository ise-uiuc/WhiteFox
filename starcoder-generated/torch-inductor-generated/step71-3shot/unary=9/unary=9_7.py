
class Model(torch.nn.Module):
    def __init__(self): # initialize a convolution layer with bias 
        super().__init__() 
        self.conv = torch.nn.Conv2d(in_channels =3, out_channels = 8, kernel_size = 1, bias = True) 
    def forward(self, x1): # apply convolution to input
        h1 = self.conv(x1)
        h2 = (3 + h1) / 6
        h3 = torch.clamp(h2, min= 0, max = 6)
        h4 = h3.clamp(min=0, max=6)
        return h4
# Inputs to the model
x1 = torch.randn(10, 3, 60, 60)
