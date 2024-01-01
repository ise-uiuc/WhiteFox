
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_trans = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1, output_padding=1)
 
    def forward(self, x):
        v1 = self.conv_trans(x)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
