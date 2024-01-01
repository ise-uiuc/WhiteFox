
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 3, (1, 2), stride=(1, 2), padding=(0, 0), dilation=(2, 2))
    def forward(self, x1):
        x2 = F.pad(x1, (1, 2, 3, 0), mode='constant', value=0)
        v1 = self.conv(x2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3,  32, 32)
