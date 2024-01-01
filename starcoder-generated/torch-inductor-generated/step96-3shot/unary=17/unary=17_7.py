
class Model(torch.nn.Module):
        def __init__(self):
                super().__init__()
                self.conv = torch.nn.ConvTranspose2d(in_channels = 4,
                                                        out_channels = 1,
                                                        kernel_size  = (1, 1))
        def forward(self, x1):
                v1 = self.conv(x1)
                v2 = torch.relu(v1)
                return v2
# Inputs to the model
x1 = torch.randn(1, 4, 16, 16)
