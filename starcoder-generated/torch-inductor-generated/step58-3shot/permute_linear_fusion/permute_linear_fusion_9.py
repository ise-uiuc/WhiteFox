
class Model(torch.nn.Module):
    shape_mismatch = False
    def __init__(self):
        super().__init__()
        self.conv_transposed = torch.nn.ConvTranspose3d(in_channels=int(35.63952636914062), out_channels=int(2.4943757248876315), kernel_size=1, stride=1, bias=True) # conv_transposed_3d [1, 35, 16, 13, 13] -> [1, 2, 16, 13, 13]
    def forward(self, x):
        # [1, 2, 16, 13, 13] -> [1, 35, 16, 13, 13]
        x1 = self.conv_transposed(x)
        # check shape
        # [1, 35, 16, 13, 13] -> [1, 2, 16, 13, 13]
        a = list(x1.size())
        b = list(x.size())
        if a!= b:
            self.shape_mismatch = True
        return x1
# Inputs to the model
x = torch.randn(1, 2, 16, 13, 13)
