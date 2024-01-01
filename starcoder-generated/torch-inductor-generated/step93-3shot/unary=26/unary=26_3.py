
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t  = torch.nn.ConvTranspose2d(3, 9, 7, stride=(2, 1), padding=(3, 2), bias=False)
        self.prelu   = torch.nn.PReLU()  
    def forward(self, x):
        z2 = self.conv_t(x)
        z3 = self.prelu(z2)
        z5 = z3 > 0
        z6 = z3 * -0.477
        z7 = torch.where(z5, z3, z6)
        return z7
# Inputs to the model
x = torch.randn(1, 3, 16, 16)
