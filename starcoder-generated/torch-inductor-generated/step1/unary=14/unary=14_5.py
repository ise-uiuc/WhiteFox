
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 4, 2, stride=1, padding=0, dilation=1, output_padding=0)
 
    def forward(self, x):
        v1 = self.conv_t(x)
        return v1 * torch.sigmoid(v1)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2, 4, 4)
