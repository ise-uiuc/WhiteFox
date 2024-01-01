
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.convt2d = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=5, stride=3, padding=0)
    def forward(self, x1):
        v1 = self.convt2d(x1)
        v2 = self.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 32, 1)
