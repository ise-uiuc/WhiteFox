
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.m0 = nn.Upsample(scale_factor = 2, mode = 'linear')
        self.m1 = nn.ConvTranspose3d(2, 3, 2, stride = 1, padding = 2, output_padding = 1)
        self.m2 = nn.Sigmoid()
    def forward(self, x):
        x = self.m0(x)
        x = self.m1(x)
        return x
# Inputs to the model
x = torch.randn(1, 2, 4, 4, 4)
