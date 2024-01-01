
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(8, 3, 3, stride=(2, 3), output_padding=0, padding=(97, 97), groups=1, dilation=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
