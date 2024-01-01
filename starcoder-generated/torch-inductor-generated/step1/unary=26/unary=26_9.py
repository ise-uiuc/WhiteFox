
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 5, 4, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv_transpose(x)
        v2 = v1 > 0
        v3 = v2 * v1
        v4 = v1 * negative_slope
        v5 = torch.where(v2, v1, v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2, 4, 4)
