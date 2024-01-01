
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 8, 16, stride=1, padding=1, bias=False)
        self.activation = torch.nn.PReLU(num_parameters=1, init=0.02)
    def forward(self, x1):
        x2 = self.conv_t((x1))
        return self.activation((x2))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
