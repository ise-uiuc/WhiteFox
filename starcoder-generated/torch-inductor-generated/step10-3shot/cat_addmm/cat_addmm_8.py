
class WeightedCat(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=-1):
        super().__init__()
        if stride == -1:
            self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size, stride, bias=False)
        else:
            self.conv = torch.nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, bias=False)
 
    def forward(self, inp, mat1, mat2):
        v1 = self.conv(inp)
        v2 = torch.addmm(inp, v1, mat1, beta=0.5, alpha=0.5)
        v3 = torch.addmm(v2, mat2, mat1)
        return v3

# Initializing the model
m = WeightedCat(3, 8, 3, stride=2)

# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(8, 3, 3, 3)
x3 = torch.randn(8, 3, 3, 3)
