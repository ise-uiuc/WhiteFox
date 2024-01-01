
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        return self.conv(x, out_channels=4, output_size=(16, 16), padding=[[1, 1, 1, 1]])

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
