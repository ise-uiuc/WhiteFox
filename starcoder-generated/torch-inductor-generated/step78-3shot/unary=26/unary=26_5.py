
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(24, 3, 3, stride=1, padding=1, out_channels=3)
    def forward(self, i1):
        i2 = self.conv_t(i1) 
        i3 = i2 > 0
        i4 = i2 * -0.299
        i5 = torch.where(i3, i2, i4)
        return torch.nn.functional.dropout(i5, training=True, inplace=False, p=0.200, )
# Inputs to the model
i1 = torch.randn(25, 24, 56, 131)
