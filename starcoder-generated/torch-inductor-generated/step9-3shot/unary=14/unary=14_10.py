
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transpose1 = torch.nn.ConvTranspose1d(in_channels=256,
                                                   out_channels=512,
                                                   kernel_size=2,
                                                   stride=2,
                                                   padding=0
                                                   )
    def forward(self, x1):
        v1 = self.transpose1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 256, 16)
