
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1, output_padding=1)
        self.negative_slope = negative_slope
 
    def forward(self, x):
        v1 = self.convt(x)
        v2 = torch.gt(v1, 0)
        v3 = torch.where(v2, v1, v1 * self.negative_slope)
        return v3

# Initializing the model
m = Model(negative_slope=-0.30000000000000004)

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
