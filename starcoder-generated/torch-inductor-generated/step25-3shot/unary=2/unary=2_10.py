
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=0),
            nn.Tanh(),
            nn.BatchNorm2d(6)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.BatchNorm2d(16)
        )
        self.transf_block = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=1, padding=0, output_padding=0)
    def forward(self, x1):
        x1 = self.block1(x1)
        x1 = self.block2(x1)
        x1 = self.transf_block(x1)
        x1 = F.pad(x1, [0, 0, 0, 0, 1, 1])
        return x1
# Inputs to the model
x1 = torch.randn(1, 1, 48, 48)
