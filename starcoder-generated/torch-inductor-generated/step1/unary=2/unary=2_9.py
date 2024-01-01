
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(in_channels=24, out_channels=11, kernel_size=(3, 7), padding=(1, 3), output_padding=(0, 1))
 
    def forward(self, input):
        v1 = self.conv(input)
        v2 = v1 * 0.5
        v3 = v1 * 0.7978845608028654
        v4 = torch.tanh(v3)
        v5 = v4 * 0.044715
        v6 = torch.mul(v2, v1)
        v7 = v6 * v2
        v8 = v5 * v7
        o = v2 + v8
        return o

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 24, 64, 64)
