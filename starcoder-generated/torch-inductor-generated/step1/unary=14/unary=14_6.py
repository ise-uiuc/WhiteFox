
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(8, 3, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8, 64, 64)
