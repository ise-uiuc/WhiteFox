
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transpose1 = nn.ConvTranspose2d(in_channels=34, out_channels=26, kernel_size=5)
        self.transpose2 = nn.ConvTranspose2d(in_channels=26, out_channels=21, kernel_size=5)
    def forward(self, x1):
        x1 = self.transpose1(x1)
        x1 = nn.ReLU()(x1)
        x1 = self.transpose2(x1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 34, 16, 16)
