
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_0 = torch.nn.Conv2d(in_channels=3, out_channels=14, kernel_size=5, stride=[2, 2], padding=[3, 3])
        self.relu_2 = torch.nn.LeakyReLU(negative_slope=0.49084136, inplace=True)
    def forward(self, x):
        negative_slope = 0.4818935
        v1 = self.conv2d_0(x)
        v2 = self.relu_2(v1 + 0.27295514)
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
    
# Inputs to the model
x1 = torch.randn(30, 3, 47, 91)
