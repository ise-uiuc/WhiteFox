
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=2)
 
    def forward(self, x):
        v7 = self.conv_t(x)
        v8 = torch.sigmoid(v7)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
