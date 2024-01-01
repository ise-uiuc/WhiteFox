
# This is a pointwise-conv11 model using Conv2d API. 
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=11, padding=5,
                                   groups=1)
    def forward(self, x1):
        v1 = self.conv(x1)  # Pointwise convolution of 11-sized kernel.
        v2 = torch.sigmoid(v1)
        v3 = torch.sigmoid(v2)  # Final two sigmoid are just for noise.
        return v3
x1 = torch.randn(1, 1, 240, 320)
