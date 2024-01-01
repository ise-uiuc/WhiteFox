
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1, output_padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return v2   
# Initializing the model
m1 = Model1()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
__output1__ = m1(x1)

